#ifndef CIR_ARM_H
#define CIR_ARM_H

namespace cir {

/// The ABI kind for ARM targets.
enum class ARMABIKind {
  APCS = 0,
  AAPCS,
  AAPCS_VFP,
  AAPCS16_VFP,
};

} // namespace cir

#endif // CIR_ARM_H
