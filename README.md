# YADE_oedo_acid
Study of an oedometric test with acid injection (dissolution of the grain matter) with the Open-Source software YADE (https://yade-dem.org/doc/).

Different files.

- like_cha_santamrina.py
    Try to reproduce and understand Cha & Santamarina 2014
- AcidOedo_Sand.py
    Investigate oedometer test under acid injection
- AcidOedo_Rock.py
    Investigate oedometer test under acid injection with cohesive grains
- AcidOedo_Rock_batch.py
    Launch parametric study with AcidOedo_Rock.py
- AcidOedo_Calcarenite.py
    Investigate oedometer test under acid injection with cohesive grain (dissolve grain and bond)

# like_cha_santamarina.py
Description in progress

# AcidOedo_Sand.py
Description in progress

# AcidOedo_Rock.py

Here a granular material is generated composed of particles and bonds between them. The dissolution occurs only at the level of the bonds. The surface of them is decreased by a constant value. Different macro parameters are tracked during the dissolution as k_0 (=\sigma_II/\sigma_I, the vertical (I) and lateral (II) pressures applied on the wall of the box).

The parameters of the simulation are set in the part "User".<br>
The description of the algorithm is presented in the following paper:<br>
Sac-Morane A, Veveakis M, Rattez H (????) ??. ??. ??:??. https://doi.org.org/?

# AcidOedo_Rock_batch.py
Same as AcidOedo_Rock.py but with yade-batch (parametric study).<br>
Work with AcidOedo_Rock.table.<br>

Different folders must be present: data and plot

# AcidOedo_Calcarenite.py
Description in progress
