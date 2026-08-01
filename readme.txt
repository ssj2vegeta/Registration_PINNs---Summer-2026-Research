Point Set Registration with PINNs

This is a research fork of Biomechanics-informed Non-rigid Medical Image Registration with Elasticity Theories" 
I am working on this codebase as a research intern under Dr. Yipeng Hu at UCL.

Current contributions:
- Toy dataset validation using sphere deformations by solving physics ODEs
  as ground truth for physics validation
- Architecture ablation study — testing robustness of stress and 
  displacement networks to feature dimension changes
- Extending the physics constraints from linear to nonlinear elasticity
- Bug fix in pinn_loss_lame_vector_cuda1_normalised (governing_equation5 
  used Sxy instead of Sxz)

Results and findings will be documented as work progresses.

Link to paper: https://discovery.ucl.ac.uk/id/eprint/10222289/1/Biomechanics-informed_Non-rigid_Medical_Image_Registration_with_Elasticity_Theories.pdf
