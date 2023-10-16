#-------------------------------------------------------------------------------
#Librairies
#-------------------------------------------------------------------------------

from yade import pack, plot, export
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import time
import math
import random
import pickle
from pathlib import Path

#-------------------------------------------------------------------------------
#User
#-------------------------------------------------------------------------------

# PSD
n_grains = 3000
L_r = []

# Particles
rMean = 0.000100  # m
rRelFuzz = .25

# Box
Dz_on_Dx = 1 # ratio Dz / Dxy
Dz = 0.0028 # m
Dx = Dz/Dz_on_Dx
Dy = Dx

# IC
n_steps_ic = 100

# Top wall
P_load = 1e7 # Pa
kp = 1e-9 # m.N-1

# Lateral wall
k0_target = 0.7
# same kp as top wall

# cementation
P_cementation = P_load*0.01 # Pa
# 2T   : E ( 300MPa), f_cemented (0.13), m_log (6.79), s_log (0.70)
# 2MB  : E ( 320MPa), f_cemented (0.88), m_log (7.69), s_log (0.60)
# 11BB : E ( 760MPa), f_cemented (0.98), m_log (8.01), s_log (0.88)
# 13BT : E ( 860MPa), f_cemented (1.00), m_log (8.44), s_log (0.92)
# 13MB : E (1000MPa), f_cemented (1.00), m_log (8.77), s_log (0.73)
type_cementation = '13MB' # only for the report
f_cemented = 1. # -
m_log = 8.77 # -
s_log = 0.73 # -
YoungModulus = 1000e6
considerYoungReduction = True
tensileCohesion = 2.75e6 # Pa
shearCohesion = 6.6e6 # Pa

# Dissolution
f_Sc_diss_1 = 2e-2
f_Sc_diss_2 = 5e-2
dSc_dissolved_1 = f_Sc_diss_1*np.exp(m_log)*1e-12
dSc_dissolved_2 = f_Sc_diss_2*np.exp(m_log)*1e-12
diss_level_1_2 = 0.95
f_n_bond_stop = 0
s_bond_diss = 0

# time step
factor_dt_crit = 0.6

# steady-state detection
unbalancedForce_criteria = 0.01

# Report
simulation_report_name = O.tags['d.id']+'_report.txt'
simulation_report = open(simulation_report_name, 'w')
simulation_report.write('Oedometer test under acid injection\n')
simulation_report.write('Type of sample: Rock\n')
simulation_report.write('Cementation at '+str(int(P_cementation))+' Pa\n')
simulation_report.write('Type of cementation: '+type_cementation+'\n')
simulation_report.write('Confinement at '+str(int(P_load))+' Pa\n')
simulation_report.write('Initial k0 targetted: '+str(round(k0_target,3))+'\n\n')
simulation_report.close()

#-------------------------------------------------------------------------------
#Initialisation
#-------------------------------------------------------------------------------

# clock to show performances
tic = time.perf_counter()
tic_0 = tic
iter_0 = 0

# plan simulation
if Path('plot').exists():
    shutil.rmtree('plot')
os.mkdir('plot')
if Path('data').exists():
    shutil.rmtree('data')
os.mkdir('data')
if Path('vtk').exists():
    shutil.rmtree('vtk')
os.mkdir('vtk')
if Path('save').exists():
    shutil.rmtree('save')
os.mkdir('save')

# define wall material (no friction)
O.materials.append(CohFrictMat(young=YoungModulus, poisson=0.25, frictionAngle=0, density=2650, isCohesive=False, momentRotationLaw=False))
# create box and grains
O.bodies.append(aabbWalls([Vector3(0,0,0),Vector3(Dx,Dy,Dz)], thickness=0.,oversizeFactor=1))
# a list of 6 boxes Bodies enclosing the packing, in the order minX, maxX, minY, maxY, minZ, maxZ
# extent the plates
O.bodies[0].shape.extents = Vector3(0,1.5*Dy/2,1.5*Dz/2)
O.bodies[1].shape.extents = Vector3(0,1.5*Dy/2,1.5*Dz/2)
O.bodies[2].shape.extents = Vector3(1.5*Dx/2,0,1.5*Dz/2)
O.bodies[3].shape.extents = Vector3(1.5*Dx/2,0,1.5*Dz/2)
O.bodies[4].shape.extents = Vector3(1.5*Dx/2,1.5*Dy/2,0)
O.bodies[5].shape.extents = Vector3(1.5*Dx/2,1.5*Dy/2,0)
# global names
lateral_plate = O.bodies[1]
upper_plate = O.bodies[-1]

# define grain material
O.materials.append(CohFrictMat(young=YoungModulus, poisson=0.25, frictionAngle=atan(0.05), density=2650,\
                               isCohesive=True, normalCohesion=tensileCohesion, shearCohesion=shearCohesion,\
                               momentRotationLaw=True, alphaKr=0, alphaKtw=0))
# frictionAngle, alphaKr, alphaKtw are set to 0 during IC. The real value is set after IC.
frictionAngleReal = radians(20)
alphaKrReal = 0.5
alphaKtwReal = 0.5

# generate grain
for i in range(n_grains):
    radius = random.uniform(rMean*(1-rRelFuzz),rMean*(1+rRelFuzz))
    center_x = random.uniform(0+radius/n_steps_ic, Dx-radius/n_steps_ic)
    center_y = random.uniform(0+radius/n_steps_ic, Dy-radius/n_steps_ic)
    center_z = random.uniform(0+radius/n_steps_ic, Dz-radius/n_steps_ic)
    O.bodies.append(sphere(center=[center_x, center_y, center_z], radius=radius/n_steps_ic))
    # can use b.state.blockedDOFs = 'xyzXYZ' to block translation of rotation of a body
    L_r.append(radius)
O.tags['Step ic'] = '1'

# yade algorithm
O.engines = [
        ForceResetter(),
        # sphere, wall
        InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Box_Aabb()]),
        InteractionLoop(
                # need to handle sphere+sphere and sphere+wall
                # Ig : compute contact point. Ig2_Sphere (3DOF) or Ig2_Sphere6D (6DOF)
                # Ip : compute parameters needed
                # Law : compute contact law with parameters from Ip
                [Ig2_Sphere_Sphere_ScGeom6D(), Ig2_Box_Sphere_ScGeom6D()],
                [Ip2_CohFrictMat_CohFrictMat_CohFrictPhys()],
                [Law2_ScGeom6D_CohFrictPhys_CohesionMoment(always_use_moment_law=True)]
        ),
        NewtonIntegrator(gravity=(0, 0, 0), damping=0.001, label = 'Newton'),
        PyRunner(command='checkUnbalanced_ir_ic()', iterPeriod = 200, label='checker')
]
# time step
O.dt = factor_dt_crit * PWaveTimeStep()

#-------------------------------------------------------------------------------

def checkUnbalanced_ir_ic():
    '''
    Increase particle radius until a steady-state is found.
    '''
    # check grains is in the box
    L_to_erase = []
    for b in O.bodies:
        if isinstance(b.shape, Sphere):
            if b.state.pos[0] < 0 or Dx < b.state.pos[0]: # x-axis
                L_to_erase.append(b.id)
            elif b.state.pos[1] < 0 or Dy < b.state.pos[1]: # y-axis
                L_to_erase.append(b.id)
            elif b.state.pos[2] < 0 or Dz < b.state.pos[2]: # z-axis
                L_to_erase.append(b.id)
    for id_to_erase in L_to_erase:
        O.bodies.erase(id_to_erase)
        print("Body",id_to_erase,'erased')
    # the rest will be run only if unbalanced is < .1 (stabilized packing)
    # Compute the ratio of mean summary force on bodies and mean force magnitude on interactions.
    if unbalancedForce() > .1:
        return
    if int(O.tags['Step ic']) < n_steps_ic :
        print('IC step '+O.tags['Step ic']+'/'+str(n_steps_ic)+' done')
        O.tags['Step ic'] = str(int(O.tags['Step ic'])+1)
        i_L_r = 0
        for b in O.bodies :
            if isinstance(b.shape, Sphere):
                growParticle(b.id, int(O.tags['Step ic'])/n_steps_ic*L_r[i_L_r]/b.shape.radius)
                i_L_r = i_L_r + 1
        O.dt = factor_dt_crit * PWaveTimeStep()
        return
    # plot the psd
    global L_L_psd_binsSizes, L_L_psd_binsProc
    L_L_psd_binsSizes = []
    L_L_psd_binsProc = []
    binsSizes, binsProc, binsSumCum = psd(bins=10)
    L_L_psd_binsSizes.append(binsSizes)
    L_L_psd_binsProc.append(binsProc)
    plotPSD()
    # characterize the ic algorithm
    global tic
    global iter_0
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac
    #report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("IC Generated : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.write(str(O.iter-iter_0)+' Iterations\n')
    simulation_report.write(str(n_grains)+' grains\n\n')
    simulation_report.close()
    print("\nIC Generated : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    # save
    #O.save('save/simu_ic.yade.bz2')
    # next time, do not call this function anymore, but the next one instead
    iter_0 = O.iter
    checker.command = 'checkUnbalanced_load_cementation_ic()'
    checker.iterPeriod = 500
    # control top wall
    O.engines = O.engines + [PyRunner(command='controlTopWall_ic()', iterPeriod = 1)]
    # switch on the gravity
    Newton.gravity = [0, 0, -9.81]

#-------------------------------------------------------------------------------

def controlTopWall_ic():
    '''
    Control the upper wall to applied a defined confinement force.

    The displacement of the wall depends on the force difference. A maximum value is defined.
    '''
    Fz = O.forces.f(upper_plate.id)[2]
    if Fz == 0:
        upper_plate.state.pos =  (lateral_plate.state.pos[0]/2, Dy/2, max([b.state.pos[2]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]))
    else :
        dF = Fz - P_cementation*lateral_plate.state.pos[0]*Dy
        v_plate_max = rMean*0.00005/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to top wall
        if v_try_abs < v_plate_max :
            upper_plate.state.vel = (0, 0, np.sign(dF)*v_try_abs)
        else :
            upper_plate.state.vel = (0, 0, np.sign(dF)*v_plate_max)

#-------------------------------------------------------------------------------

def checkUnbalanced_load_cementation_ic():
    addPlotData_cementation_ic()
    saveData_ic()
    # check the force applied
    if abs(O.forces.f(upper_plate.id)[2]-P_cementation*Dx*Dy)/(P_cementation*Dx*Dy) > 0.005:
        return
    if unbalancedForce() > unbalancedForce_criteria :
        return
    # characterize the ic algorithm
    global tic
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac
    #report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Pressure (Cementation) applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.write(str(n_grains)+' grains\n\n')
    simulation_report.close()
    print("\nPressure (Cementation) applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    # switch on friction, rolling resistance and twisting resistance between particles
    O.materials[-1].frictionAngle = frictionAngleReal
    O.materials[-1].alphaKr = alphaKrReal
    O.materials[-1].alphaKtw = alphaKtwReal
    # for existing contacts, clear them
    O.interactions.clear()
    # calm down particle
    for b in O.bodies:
        if isinstance(b.shape,Sphere):
            b.state.angVel = Vector3(0,0,0)
            b.state.vel = Vector3(0,0,0)
    # switch off damping
    Newton.damping = 0
    # next time, do not call this function anymore, but the next one instead
    checker.command = 'checkUnbalanced_param_ic()'

#-------------------------------------------------------------------------------

def checkUnbalanced_param_ic():
    addPlotData_cementation_ic()
    saveData_ic()
    # check the force applied
    if abs(O.forces.f(upper_plate.id)[2]-P_cementation*Dx*Dy)/(P_cementation*Dx*Dy) > 0.005:
        return
    if unbalancedForce() > unbalancedForce_criteria :
        return
    # characterize the ic algorithm
    global tic
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac
    #report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Parameters applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n\n")
    simulation_report.close()
    print("\nParameters applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    # save
    #O.save('save/'+O.tags['d.id']+'_ic.yade.bz2')
    # next time, do not call this function anymore, but the next one instead
    checker.command = 'cementation()'
    checker.iterPeriod = 10

#------------------------------------------------------------------------------

def pdf_lognormal(x,m_log,s_log):
    '''
    Return the probability of a value x with a log normal function defined by the mean m_log and the variance s_log.
    '''
    p = np.exp(-(np.log(x)-m_log)**2/(2*s_log**2))/(x*s_log*np.sqrt(2*np.pi))
    return p

#-------------------------------------------------------------------------------

def cementation():
    '''
    Generate cementation between grains.
    '''
    # generate the list of cohesive surface area and its list of weight
    x_min = 1e2 # µm2
    x_max = 4e4 # µm2
    n_x = 20000
    x_L = np.linspace(x_min, x_max, n_x)
    p_x_L = []
    for x in x_L :
        p_x_L.append(pdf_lognormal(x,m_log,s_log))
    # counter
    global counter_bond, counter_bond0, counter_bond_broken_diss, counter_bond_broken_load
    counter_bond = 0
    counter_bond_broken_diss = 0
    counter_bond_broken_load = 0
    # iterate on interactions
    for i in O.interactions:
        # only grain-grain contact can be cemented
        if isinstance(O.bodies[i.id1].shape, Sphere) and isinstance(O.bodies[i.id2].shape, Sphere) :
            # only a fraction of the contact is cemented
            if random.uniform(0,1) < f_cemented :
                counter_bond = counter_bond + 1
                # creation of cohesion
                i.phys.cohesionBroken = False
                # determine the cohesive surface
                cohesiveSurface = random.choices(x_L,p_x_L)[0]*1e-12 # µm2
                # set normal and shear adhesions
                i.phys.normalAdhesion = tensileCohesion*cohesiveSurface
                i.phys.shearAdhesion = shearCohesion*cohesiveSurface
    counter_bond0 = counter_bond
    # write in the report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write(str(counter_bond)+" contacts cemented\n\n")
    simulation_report.close()
    print('\n'+str(counter_bond)+" contacts cemented\n")

    # next time, do not call this function anymore, but the next one instead
    checker.command = 'checkUnbalanced_load_confinement_ic()'
    checker.iterPeriod = 200
    # activate Young reduction
    if considerYoungReduction :
        O.engines = O.engines[:-1] + [PyRunner(command='YoungReduction()', iterPeriod = 1)] + [O.engines[-1]]
    # change the vertical pressure applied
    O.engines = O.engines[:-1] + [PyRunner(command='controlTopWall()', iterPeriod = 1)]

#-------------------------------------------------------------------------------

def checkUnbalanced_load_confinement_ic():
    addPlotData_confinement_ic()
    saveData_ic()
    # check the force applied
    if abs(O.forces.f(upper_plate.id)[2]-P_load*Dx*Dy)/(P_load*Dx*Dy) > 0.005:
        return
    if unbalancedForce() > unbalancedForce_criteria :
        return
    # characterize the ic algorithm
    global tic
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac
    #report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Pressure (Confinement) applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.write(str(n_grains)+' grains\n\n')
    simulation_report.close()
    print("\nPressure (Confinement) applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")

    # next time, do not call this function anymore, but the next one instead
    checker.command = 'checkUnbalanced_load_k0_ic()'
    # control lateral wall
    O.engines = O.engines + [PyRunner(command='controlLateralWall_ic()', iterPeriod = 1, label='k0_checker')]

#-------------------------------------------------------------------------------

def controlLateralWall_ic():
    '''
    Control the lateral wall to applied a defined confinement force.

    The displacement of the wall depends on the force difference. A maximum value is defined.
    '''
    Fx = O.forces.f(lateral_plate.id)[0]
    if Fx == 0:
        lateral_plate.state.pos =  (max([b.state.pos[0]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), Dy/2, upper_plate.state.pos[2]/2)
    else :
        dF = Fx - k0_target*P_load*upper_plate.state.pos[2]*Dy
        v_plate_max = rMean*0.00002/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to top wall
        if v_try_abs < v_plate_max :
            lateral_plate.state.vel = (np.sign(dF)*v_try_abs, 0, 0)
        else :
            lateral_plate.state.vel = (np.sign(dF)*v_plate_max, 0, 0)

#-------------------------------------------------------------------------------

def DoNotControlLateralWall_ic():
    '''
    Switch off the control of the lateral wall.
    '''
    lateral_plate.state.vel = (0,0,0)
    O.engines = O.engines[:-1]

#-------------------------------------------------------------------------------

def checkUnbalanced_load_k0_ic():
    addPlotData_confinement_ic()
    saveData_ic()
    # check the force applied
    if abs(O.forces.f(upper_plate.id)[2]  - P_load*lateral_plate.state.pos[0]*Dy)/(P_load*lateral_plate.state.pos[0]*Dy) > 0.005 or\
    abs(O.forces.f(lateral_plate.id)[0] - k0_target*P_load*upper_plate.state.pos[2]*Dy)/(k0_target*P_load*upper_plate.state.pos[2]*Dy) > 0.005:
        return
    if unbalancedForce() > unbalancedForce_criteria :
        return

    # characterize the ic algorithm
    global tic, iter_0
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac
    #report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Pressure (k0) applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.write('IC generation ends\n')
    simulation_report.write(str(O.iter-iter_0)+' Iterations\n')
    simulation_report.write(str(n_grains)+' grains\n\n')
    simulation_report.close()
    print("\nPressure (k0) applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")

    # reset plot (IC done, simulation starts)
    plot.reset()
    # switch off control lateral wall
    O.engines = O.engines[:-1]+[PyRunner(command='DoNotControlLateralWall_ic()', iterPeriod = 1)]
    # save new reference position for upper wall
    upper_plate.state.refPos = upper_plate.state.pos
    # next time, do not call this function anymore, but the next one instead
    iter_0 = O.iter
    checker.command = 'checkUnbalanced()'
    checker.iterPeriod = 500
    # label step
    O.tags['Current Step']='0'
    # trackers
    global L_unbalanced_ite, L_k0_ite, L_confinement_ite, L_count_bond
    L_unbalanced_ite = []
    L_k0_ite = []
    L_confinement_ite = []
    L_count_bond = []

#-------------------------------------------------------------------------------

def addPlotData_cementation_ic():
    """
    Save data in plot.
    """
    # add forces applied on wall x and z
    sz = O.forces.f(upper_plate.id)[2]/(lateral_plate.state.pos[0]*Dy)
    sx = O.forces.f(lateral_plate.id)[0]/(Dy*upper_plate.state.pos[2])
    # add data
    plot.addData(i=O.iter-iter_0, porosity=porosity(), coordination=avgNumInteractions(), unbalanced=unbalancedForce(),\
                 Sx=sx, Sz=sz, conf_verified=sz/P_cementation*100, n_bond=0,\
                 vert_strain=100*(upper_plate.state.pos[2]-upper_plate.state.refPos[2])/upper_plate.state.refPos[2], lat_strain=100*(lateral_plate.state.pos[0]-lateral_plate.state.refPos[0])/lateral_plate.state.refPos[0])

#-------------------------------------------------------------------------------

def addPlotData_confinement_ic():
    """
    Save data in plot.
    """
    # add forces applied on wall x and z
    sz = O.forces.f(upper_plate.id)[2]/(lateral_plate.state.pos[0]*Dy)
    sx = O.forces.f(lateral_plate.id)[0]/(Dy*upper_plate.state.pos[2])
    # count the number the bond
    n_bond = 0
    for i in O.interactions:
        if isinstance(O.bodies[i.id1].shape, Sphere) and isinstance(O.bodies[i.id2].shape, Sphere):
            if not i.phys.cohesionBroken :
                n_bond = n_bond + 1
    # add data
    plot.addData(i=O.iter-iter_0, porosity=porosity(), coordination=avgNumInteractions(), unbalanced=unbalancedForce(),\
                 Sx=sx, Sz=sz, conf_verified=sz/P_load*100, n_bond=n_bond,\
                 vert_strain=100*(upper_plate.state.pos[2]-upper_plate.state.refPos[2])/upper_plate.state.refPos[2], lat_strain=100*(lateral_plate.state.pos[0]-lateral_plate.state.refPos[0])/lateral_plate.state.refPos[0])

#-------------------------------------------------------------------------------

def saveData_ic():
    """
    Save data in .txt file during the ic.
    """
    plot.saveDataTxt('data/IC_'+O.tags['d.id']+'.txt')
    # post-proccess
    L_sigma_x = []
    L_sigma_z = []
    L_confinement = []
    L_coordination = []
    L_unbalanced = []
    L_ite  = []
    L_vert_strain = []
    L_lat_strain = []
    L_porosity = []
    L_n_bond = []
    file = 'data/IC_'+O.tags['d.id']+'.txt'
    data = np.genfromtxt(file, skip_header=1)
    file_read = open(file, 'r')
    lines = file_read.readlines()
    file_read.close()
    if len(lines) >= 3:
        for i in range(len(data)):
            L_sigma_x.append(abs(data[i][0]))
            L_sigma_z.append(abs(data[i][1]))
            L_confinement.append(data[i][2])
            L_coordination.append(data[i][3])
            L_ite.append(data[i][4])
            L_lat_strain.append(data[i][5])
            L_n_bond.append(data[i][6])
            L_porosity.append(data[i][7])
            L_unbalanced.append(data[i][8])
            L_vert_strain.append(data[i][9])

        # plot
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(20,10),num=1)

        ax1.plot(L_ite, L_sigma_x, label = r'$\sigma_x$')
        ax1.plot(L_ite, L_sigma_z, label = r'$\sigma_z$')
        ax1.legend()
        ax1.set_title('Stresses (Pa)')

        ax2.plot(L_ite, L_unbalanced, 'b')
        ax2.set_ylabel('Unbalanced (-)', color='b')
        ax2.set_ylim(ymin=0, ymax=2*unbalancedForce_criteria)
        ax2b = ax2.twinx()
        ax2b.plot(L_ite, L_confinement, 'r')
        ax2b.set_ylabel('Confinement (%)', color='r')
        ax2b.set_title('Steady-state indices')

        ax3.plot(L_ite, L_n_bond)
        ax3.set_title('Number of bond (-)')

        ax4.plot(L_ite, L_lat_strain, label=r'$\epsilon_x$ (%)')
        ax4.plot(L_ite, L_vert_strain, label=r'$\epsilon_v$ (%)')
        ax4.legend()
        ax4.set_title('Strains (%)')

        ax5.plot(L_ite, L_porosity)
        ax5.set_title('Porosity (-)')

        ax6.plot(L_ite, L_coordination)
        ax6.set_title('Coordination number (-)')

        plt.savefig('plot/IC_'+O.tags['d.id']+'.png')

        plt.close()

#-------------------------------------------------------------------------------
#Load
#-------------------------------------------------------------------------------

def YoungReduction():
    '''
    Reduce the Young modulus with the number of bond dissolved.

    E = (E0-E1)*f_diss + E1
    '''
    f_bond_diss = (counter_bond_broken_diss+counter_bond_broken_load)/counter_bond0
    NewYoungModulus = (YoungModulus-80e6)*(1-f_bond_diss) + 80e6
    # update material
    for mat in O.materials :
        mat.young = NewYoungModulus
    # update the interactions
    for inter in O.interactions :
        if isinstance(O.bodies[inter.id1].shape, Sphere) and isinstance(O.bodies[inter.id2].shape, Sphere):
            inter.phys.kn = NewYoungModulus*(O.bodies[inter.id1].shape.radius*2*O.bodies[inter.id2].shape.radius*2)/(O.bodies[inter.id1].shape.radius*2+O.bodies[inter.id2].shape.radius*2)
            inter.phys.ks = 0.25*NewYoungModulus*(O.bodies[inter.id1].shape.radius*2*O.bodies[inter.id2].shape.radius*2)/(O.bodies[inter.id1].shape.radius*2+O.bodies[inter.id2].shape.radius*2) # 0.25 is the Poisson ratio
            inter.phys.kr = inter.phys.ks*alphaKrReal*O.bodies[inter.id1].shape.radius*O.bodies[inter.id2].shape.radius
            inter.phys.ktw = inter.phys.ks*alphaKtwReal*O.bodies[inter.id1].shape.radius*O.bodies[inter.id2].shape.radius
        else : # Sphere-Wall contact
            if isinstance(O.bodies[inter.id1].shape, Sphere):
                grain = O.bodies[inter.id1]
            else:
                grain = O.bodies[inter.id2]
            # diameter of the wall is equivalent of the diameter of the sphere
            inter.phys.kn = NewYoungModulus*(grain.shape.radius*2*grain.shape.radius*2)/(grain.shape.radius*2+grain.shape.radius*2)
            inter.phys.ks = 0.25*NewYoungModulus*(grain.shape.radius*2*grain.shape.radius*2)/(grain.shape.radius*2+grain.shape.radius*2) # 0.25 is the Poisson ratio
            # no moment/twist for sphere-wall

#-------------------------------------------------------------------------------

def controlTopWall():
    '''
    Control the upper wall to applied a defined confinement force.

    The displacement of the wall depends on the force difference. A maximum value is defined.
    '''
    Fz = O.forces.f(upper_plate.id)[2]
    if Fz == 0:
        upper_plate.state.pos =  (lateral_plate.state.pos[0]/2, Dy/2, max([b.state.pos[2]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]))
    else :
        dF = Fz - P_load*lateral_plate.state.pos[0]*Dy
        v_plate_max = rMean*0.00005/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to top wall
        if v_try_abs < v_plate_max :
            upper_plate.state.vel = (0, 0, np.sign(dF)*v_try_abs)
        else :
            upper_plate.state.vel = (0, 0, np.sign(dF)*v_plate_max)

#-------------------------------------------------------------------------------

def count_bond():
    '''
    Count the bond
    '''
    counter_bond = 0
    for i in O.interactions:
        if isinstance(O.bodies[i.id1].shape, Sphere) and isinstance(O.bodies[i.id2].shape, Sphere):
            if not i.phys.cohesionBroken :
                counter_bond = counter_bond + 1
    return counter_bond

#-------------------------------------------------------------------------------

def checkUnbalanced():
    """
    Look for the steady state during the loading phase.
    """
    # track and plot unbalanced
    global L_unbalanced_ite, L_k0_ite, L_confinement_ite, L_count_bond
    L_unbalanced_ite.append(unbalancedForce())
    if O.forces.f(upper_plate.id)[2] != 0:
        k0 = abs(O.forces.f(lateral_plate.id)[0]/(upper_plate.state.pos[2]*Dy)*(lateral_plate.state.pos[0]*Dy)/O.forces.f(upper_plate.id)[2])
    else :
        k0 = 0
    L_k0_ite.append(k0)
    L_confinement_ite.append(O.forces.f(upper_plate.id)[2]/(P_load*lateral_plate.state.pos[0]*Dy)*100)
    L_count_bond.append(count_bond())

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,9),num=1)

    ax1.plot(L_unbalanced_ite)
    ax1.set_title('unbalanced force (-)')

    ax2.plot(L_k0_ite)
    ax2.set_title(r'$k_0$ (-)')

    ax3.plot(L_confinement_ite)
    ax3.set_title('confinement (%)')

    ax4.plot(L_count_bond)
    ax4.set_title('Number of bond (-)')

    fig.savefig('plot/tracking_ite.png')
    plt.close()

    if (unbalancedForce() < unbalancedForce_criteria) and \
       (abs(O.forces.f(upper_plate.id)[2]-P_load*lateral_plate.state.pos[0]*Dy) < 0.01*P_load*lateral_plate.state.pos[0]*Dy):

        # reset trackers
        L_unbalanced_ite = []
        L_k0_ite = []
        L_confinement_ite = []
        L_count_bond = []

        if counter_bond0*f_n_bond_stop < counter_bond:
            dissolve()
        else :
            stopLoad()

#-------------------------------------------------------------------------------

def dissolve():
    """
    Dissolve bond with a constant surface reduction.
    """
    O.tags['Current Step'] = str(int(O.tags['Current Step'])+1)
    # count the number of bond
    global counter_bond, counter_bond_broken_diss, counter_bond_broken_load, s_bond_diss
    counter_bond = 0
    counter_bond_broken = 0
    # iterate on interactions
    for i in O.interactions:
        # only grain-grain contact can be cemented
        if isinstance(O.bodies[i.id1].shape, Sphere) and isinstance(O.bodies[i.id2].shape, Sphere) :
            if not i.phys.cohesionBroken :
                counter_bond = counter_bond + 1
                # set normal and shear adhesions
                if (counter_bond_broken_diss+counter_bond_broken_load)/counter_bond0 < diss_level_1_2 :
                    i.phys.normalAdhesion = i.phys.normalAdhesion - tensileCohesion*dSc_dissolved_1
                    i.phys.shearAdhesion = i.phys.shearAdhesion - shearCohesion*dSc_dissolved_1
                else :
                    i.phys.normalAdhesion = i.phys.normalAdhesion - tensileCohesion*dSc_dissolved_2
                    i.phys.shearAdhesion = i.phys.shearAdhesion - shearCohesion*dSc_dissolved_2
                if i.phys.normalAdhesion <= 0 or i.phys.shearAdhesion <=0 :
                    # bond brokes
                    counter_bond = counter_bond - 1
                    counter_bond_broken = counter_bond_broken + 1
                    i.phys.cohesionBroken = True
                    i.phys.normalAdhesion = 0
                    i.phys.shearAdhesion = 0
    # update bond surface dissolved tracker
    if (counter_bond_broken_diss+counter_bond_broken_load)/counter_bond0 < diss_level_1_2 :
        s_bond_diss = s_bond_diss + dSc_dissolved_1
    else :
        s_bond_diss = s_bond_diss + dSc_dissolved_2
    # update the counter of bond dissolved during the dissolution step
    counter_bond_broken_diss = counter_bond_broken_diss + counter_bond_broken
    counter_bond_broken_load = (counter_bond0-counter_bond) - counter_bond_broken_diss
    # save at the end
    saveData()
    # update time step
    O.dt = factor_dt_crit * PWaveTimeStep()

#-------------------------------------------------------------------------------

def stopLoad():
    """
    Close simulation.
    """
    # save at the converged iteration
    saveData()
    # close yade
    O.pause()
    # characterize the dem step
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    #report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Oedometric test : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.close()
    print("Oedometric test : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    # characterize the last DEM step and the simulation
    hours = (tac-tic_0)//(60*60)
    minutes = (tac-tic_0 -hours*60*60)//(60)
    seconds = int(tac-tic_0 -hours*60*60 -minutes*60)
    #report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Simulation time : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n\n")
    simulation_report.close()
    print("\nSimulation time : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")

    # give final values
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("k0 initial: "+str(round(k0_target,3))+" / k0 final: "+str(round(abs(O.forces.f(lateral_plate.id)[0]/(upper_plate.state.pos[2]*Dy)*(lateral_plate.state.pos[0]*Dy)/O.forces.f(upper_plate.id)[2]),3))+"\n")
    simulation_report.write(str(int(counter_bond_broken_diss))+" bonds broken during dissolution steps ("+str(int(counter_bond_broken_diss/counter_bond0*100))+" % of initial number of bonds)\n")
    simulation_report.write(str(int(counter_bond_broken_load))+" bonds broken during loading steps ("+str(int(counter_bond_broken_load/counter_bond0*100))+" % of initial number of bonds)\n")
    simulation_report.close()
    print("k0 initial:",round(k0_target,3)," / k0 final:",round(abs(O.forces.f(lateral_plate.id)[0]/(upper_plate.state.pos[2]*Dy)*(lateral_plate.state.pos[0]*Dy)/O.forces.f(upper_plate.id)[2]),3))
    print(int(counter_bond_broken_diss),"bonds broken during dissolution steps ("+str(int(counter_bond_broken_diss/counter_bond0*100)),"% of initial number of bonds)")
    print(int(counter_bond_broken_load),"bonds broken during loading steps ("+str(int(counter_bond_broken_load/counter_bond0*100)),"% of initial number of bonds)")

    # save simulation
    os.mkdir('../AcidOedo_Rock_data/'+O.tags['d.id'])
    shutil.copytree('data','../AcidOedo_Rock_data/'+O.tags['d.id']+'/data')
    shutil.copytree('plot','../AcidOedo_Rock_data/'+O.tags['d.id']+'/plot')
    shutil.copytree('save','../AcidOedo_Rock_data/'+O.tags['d.id']+'/save')
    shutil.copy('AcidOedo_Rock.py','../AcidOedo_Rock_data/'+O.tags['d.id']+'/AcidOedo_Rock.py')
    shutil.copy(O.tags['d.id']+'_report.txt','../AcidOedo_Rock_data/'+O.tags['d.id']+'/'+O.tags['d.id']+'_report.txt')

#-------------------------------------------------------------------------------

def addPlotData():
    """
    Save data in plot.
    """
    # add forces applied on wall x and z
    sz = O.forces.f(upper_plate.id)[2]/(Dy*lateral_plate.state.pos[0])
    sx = O.forces.f(lateral_plate.id)[0]/(Dy*upper_plate.state.pos[2])
    # compute the k0 = sigma_x/sigma_z, = 0 if no sigma_z
    if sz != 0:
        k0 = abs(sx/sz)
    else :
        k0 = 0
    # add data
    plot.addData(i=O.iter-iter_0, porosity=porosity(), coordination=avgNumInteractions(), unbalanced=unbalancedForce(), \
                counter_bond=counter_bond, counter_bond_broken_diss=counter_bond_broken_diss, counter_bond_broken_load=counter_bond_broken_load,\
                Sx=sx, Sz=sz, Z_plate=upper_plate.state.pos[2], conf_verified=sz/P_load*100, k0=k0,\
                w=upper_plate.state.pos[2]-upper_plate.state.refPos[2], vert_strain=100*(upper_plate.state.pos[2]-upper_plate.state.refPos[2])/upper_plate.state.refPos[2],
                s_bond_diss=s_bond_diss)

#-------------------------------------------------------------------------------

def saveData():
    """
    Save data in .txt file during the steps.
    """
    addPlotData()
    plot.saveDataTxt('data/'+O.tags['d.id']+'.txt')
    # post-proccess
    L_s_bond_diss = []
    L_k0 = []
    L_counter_bond = []
    L_counter_bond_broken_diss = []
    L_counter_bond_broken_load = []
    L_vert_strain = []
    L_porosity = []
    L_coordination = []
    file = 'data/'+O.tags['d.id']+'.txt'
    data = np.genfromtxt(file, skip_header=1)
    file_read = open(file, 'r')
    lines = file_read.readlines()
    file_read.close()
    if len(lines) >= 3:
        for i in range(len(data)):
            L_coordination.append(data[i][4])
            L_counter_bond.append(data[i][5])
            L_counter_bond_broken_diss.append(data[i][6])
            L_counter_bond_broken_load.append(data[i][7])
            L_k0.append(data[i][9])
            L_porosity.append(data[i][10])
            L_s_bond_diss.append(data[i][11])
            L_vert_strain.append(data[i][12])

        # plot
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(16,9),num=1)

        ax1.plot(L_s_bond_diss, L_k0)
        ax1.set_title(r'$k_0$ (-)')

        ax2.plot(L_s_bond_diss, L_counter_bond)
        ax2.set_title('Number of bond (-)')

        ax3.plot(L_s_bond_diss, L_counter_bond_broken_diss, label='during dissolution')
        ax3.plot(L_s_bond_diss, L_counter_bond_broken_load, label='during loading')
        ax3.set_title('Number of bonds broken (-)')
        ax3.legend()

        ax4.plot(L_s_bond_diss, L_vert_strain)
        ax4.set_title(r'$\epsilon_v$ (%)')

        ax5.plot(L_s_bond_diss, L_porosity)
        ax5.set_title('Porosity (-)')

        ax6.plot(L_s_bond_diss, L_coordination)
        ax6.set_title('Coordination (-)')

        plt.suptitle(r'Trackers - bond surface reduction (m$^2$)')
        plt.savefig('plot/'+O.tags['d.id']+'.png')
        plt.close()

#-------------------------------------------------------------------------------

def plotPSD():
    """
    This function can be called to plot the evolution of the psd.
    """
    plt.figure(1, figsize=(16,9))
    for i_psd in range(len(L_L_psd_binsSizes)):
        binsSizes = L_L_psd_binsSizes[i_psd]
        binsProc = L_L_psd_binsProc[i_psd]
        plt.plot(binsSizes, binsProc)
    plt.title('Particle Size Distribution')
    plt.savefig('plot/PSD.png')
    plt.close()

#-------------------------------------------------------------------------------

def whereAmI():
    """
    This function can be called during a simulation to give information to the user.
    """
    print()
    print("Where am I ?")
    print()
    if 'Current Step' in O.tags.keys():
        print('Iteration',O.iter)
        print(O.tags['Current Step'],'dissolutions done :')
        print('Sample description :')
        print('\tNumber of bonds =', int(counter_bond),'(/)',int(counter_bond0))
        print('\tepsilon_v =', round(100*(upper_plate.state.pos[2]-upper_plate.state.refPos[2])/upper_plate.state.refPos[2],2),'(%)')
        print('\tPorosity =', round(porosity(),3),'(-)')
        print('\tCoordination =', round(avgNumInteractions(),1),'(-)')
    else :
        print('Initialisation')

#-------------------------------------------------------------------------------
# start simulation
#-------------------------------------------------------------------------------

O.run()
waitIfBatch()
