//===- ARM.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Target/ARM.h"
#include "ABIInfoImpl.h"
#include "LowerFunctionInfo.h"
#include "LowerTypes.h"
#include "TargetInfo.h"
#include "TargetLoweringInfo.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using ARMABIKind = cir::ARMABIKind;
using ABIArgInfo = cir::ABIArgInfo;
using MissingFeatures = cir::MissingFeatures;

namespace cir {

//===----------------------------------------------------------------------===//
// ARM ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class ARMABIInfo : public ABIInfo {
  ARMABIKind Kind;

public:
  ARMABIInfo(LowerTypes &CGT, ARMABIKind Kind) : ABIInfo(CGT), Kind(Kind) {}

private:
  ARMABIKind getABIKind() const { return Kind; }

  ABIArgInfo classifyReturnType(mlir::Type RetTy, bool IsVariadic) const;
  ABIArgInfo classifyArgumentType(mlir::Type RetTy, bool IsVariadic,
                                  unsigned CallingConvention) const;

  void computeInfo(LowerFunctionInfo &FI) const override {
    if (!cir::classifyReturnType(getCXXABI(), FI, *this))
      FI.getReturnInfo() =
          classifyReturnType(FI.getReturnType(), FI.isVariadic());

    for (auto &it : FI.arguments())
      it.info = classifyArgumentType(it.type, FI.isVariadic(),
                                     FI.getCallingConvention());
  }
};

class ARMTargetLoweringInfo : public TargetLoweringInfo {
public:
  ARMTargetLoweringInfo(LowerTypes &LT, ARMABIKind Kind)
      : TargetLoweringInfo(std::make_unique<ARMABIInfo>(LT, Kind)) {
    cir_cconv_assert(!MissingFeatures::swift());
  }

  unsigned getTargetAddrSpaceFromCIRAddrSpace(
      cir::AddressSpaceAttr addressSpaceAttr) const override {
    using Kind = cir::AddressSpaceAttr::Kind;
    switch (addressSpaceAttr.getValue()) {
    case Kind::offload_private:
    case Kind::offload_local:
    case Kind::offload_global:
    case Kind::offload_constant:
    case Kind::offload_generic:
      return 0;
    default:
      cir_cconv_unreachable("Unknown CIR address space for this target");
    }
  }
};

} // namespace

ABIArgInfo ARMABIInfo::classifyReturnType(mlir::Type RetTy,
                                          bool IsVariadic) const {
  // Handle void return type
  if (mlir::isa<VoidType>(RetTy))
    return ABIArgInfo::getIgnore();

  if (mlir::isa<VectorType>(RetTy)) {
    // Large vector types should be returned via memory.
    if (getContext().getTypeSize(RetTy) > 128)
      return getNaturalAlignIndirect(RetTy);

    // Check if VFP/NEON is available (AAPCS-VFP variant)
    if (!IsVariadic && getABIKind() == ARMABIKind::AAPCS_VFP) {
      uint64_t Size = getContext().getTypeSize(RetTy);
      if (Size <= 128)
        return ABIArgInfo::getDirect();
    }

    cir_cconv_assert_or_abort(!cir::MissingFeatures::vectorType(), "NYI");
  }

  if (!isAggregateTypeForABI(RetTy)) {
    if (MissingFeatures::fixedSizeIntType())
      cir_cconv_unreachable("NYI");

    if (getContext().getTypeSize(RetTy) > 64)
      return getNaturalAlignIndirect(RetTy);

    // Promote small integers
    return (isPromotableIntegerTypeForABI(RetTy) ? ABIArgInfo::getExtend(RetTy)
                                                 : ABIArgInfo::getDirect());
  }

  cir_cconv_assert(!cir::MissingFeatures::emitEmptyRecordCheck());

  // Handle aggregates
  uint64_t Size = getContext().getTypeSize(RetTy);

  // Empty records check
  if (Size == 0)
    return ABIArgInfo::getIgnore();

  // Small aggregates handling (<=4 bytes)
  if (Size <= 32) {
    if (getDataLayout().isBigEndian()) {
      // For big-endian, return in 32-bit register
      return ABIArgInfo::getDirect(
          IntType::get(LT.getMLIRContext(), 32, false));
    }
    // For little-endian, return in appropriately sized register
    return ABIArgInfo::getDirect(
        IntType::get(LT.getMLIRContext(), Size, false));
  }

  // Aggregates <= 16 bytes are returned directly in registers or on the stack.
  if (Size <= 128) {
    if (Size <= 64 && !getDataLayout().isBigEndian()) {
      // Composite types are returned in lower bits of a 64-bit register for LE,
      // and in higher bits for BE. However, integer types are always returned
      // in lower bits for both LE and BE, and they are not rounded up to
      // 64-bits. We can skip rounding up of composite types for LE, but not for
      // BE, otherwise composite types will be indistinguishable from integer
      // types.
      return ABIArgInfo::getDirect(
          cir::IntType::get(LT.getMLIRContext(), Size, false));
    }

    Size = llvm::alignTo(Size, 32); // round up to multiple of 4 bytes

    // Use register pairs for 4-byte aligned aggregates
    mlir::Type BaseTy = IntType::get(LT.getMLIRContext(), 32, false);
    return ABIArgInfo::getDirect(
        ArrayType::get(LT.getMLIRContext(), BaseTy, Size / 32));
  }

  return getNaturalAlignIndirect(RetTy);
}

ABIArgInfo ARMABIInfo::classifyArgumentType(mlir::Type Ty, bool IsVariadic,
                                            unsigned CallingConvention) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  if (mlir::isa<VectorType>(Ty)) {
    uint64_t Size = getContext().getTypeSize(Ty);
    if (Size > 128)
      return getNaturalAlignIndirect(Ty);

    // VFP/NEON vectors up to 16 bytes are passed in registers
    if (Size <= 128 && !IsVariadic && getABIKind() == ARMABIKind::AAPCS_VFP) {
      return ABIArgInfo::getDirect();
    }

    cir_cconv_assert_or_abort(!cir::MissingFeatures::vectorType(), "NYI");
  }

  if (!isAggregateTypeForABI(Ty)) {
    // NOTE(cir): Enum is IntType in CIR. Skip enum handling here.

    if (MissingFeatures::fixedSizeIntType())
      cir_cconv_unreachable("NYI");

    if (mlir::isa<mlir::FloatType>(Ty)) {
      if (getABIKind() == ARMABIKind::AAPCS_VFP && !IsVariadic)
        return ABIArgInfo::getDirect(); // Pass in VFP registers

      // For soft float ABI, promote float to double
      if (getContext().getTypeSize(Ty) == 32)
        return ABIArgInfo::getExtend(Ty);

      return ABIArgInfo::getDirect();
    }

    // Handle integer types
    if (isPromotableIntegerTypeForABI(Ty)) {
      return ABIArgInfo::getExtend(Ty);
    }

    // TODO(kumarak): Handle 64-bit integers

    return ABIArgInfo::getDirect();
  }

  uint64_t Size = getContext().getTypeSize(Ty);

  // Empty records are ignored
  if (Size == 0)
    return ABIArgInfo::getIgnore();

  // Small aggregates (up to 4 bytes) are passed in a single register
  if (Size <= 32) {
    return ABIArgInfo::getDirect(
        IntType::get(LT.getMLIRContext(), Size, false));
  }

  // Aggregates up to 4 words (16 bytes) are passed in registers
  if (Size <= 128) {
    // Round size up to register size (32 bits)
    Size = llvm::alignTo(Size, 32);

    mlir::Type BaseTy = IntType::get(LT.getMLIRContext(), 32, false);
    return ABIArgInfo::getDirect(
        ArrayType::get(LT.getMLIRContext(), BaseTy, Size / 32));
  }

  return getNaturalAlignIndirect(Ty, /*ByVal=*/false);
}

std::unique_ptr<TargetLoweringInfo>
createARMTargetLoweringInfo(LowerModule &CGM, ARMABIKind Kind) {
  return std::make_unique<ARMTargetLoweringInfo>(CGM.getTypes(), Kind);
}

} // namespace cir
