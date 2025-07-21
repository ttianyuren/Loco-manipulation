# Changelog – TIAGo++ Description

All notable changes to this model will be documented in this file.

## [06/09/2024]

- Initial release.

## [11/06/2025]

- **Added collision geometries** using meshes and primitives:
  - **Arms and base**: Collision meshes sourced from [tiago_description/meshes](https://github.com/pal-robotics/tiago_robot/tree/humble-devel/tiago_description/meshes)
  - **Torso**: Collision meshes from [tiago_dual_description/meshes](https://github.com/pal-robotics/tiago_dual_robot/tree/humble-devel/tiago_dual_description/meshes)
  - **Gripper**: Collision meshes from [pal_gripper_description/meshes](https://github.com/pal-robotics/pal_gripper/tree/humble-devel/pal_gripper_description/meshes)

- **Collision approximation strategy**:
  - Arms and gripper approximated using **capsules** generated from the provided collision meshes. This is faster to implement than manually tuning capsule sizes, though less precise.
  - Torso and base approximated using **boxes**, which provide a more suitable fit for these components.

- **Rationale**:
  - Using distance calculations directly from meshes led to issues, see [#1784](https://github.com/google-deepmind/mujoco/issues/1784#issuecomment-2303899197).
  - Simplified primitive shapes offer more stable and efficient collision handling in simulation.

## [14/06/2025]

- **Replaced box collision geometries with capsules**:
  - Distance calculations between box-to-box geometries in MuJoCo have a bug [mujoco#2710](https://github.com/google-deepmind/mujoco/issues/2710), we therefore approximate the collision boxes by capsules. This should be removed once the bug has been fixed has capsules don't provide a good approximation.
