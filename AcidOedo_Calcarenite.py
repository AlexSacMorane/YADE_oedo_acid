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
P_cementation = 1e4 # Pa
F_cementation = P_cementation*Dx*Dy # N
P_load = 1e5 # Pa
F_load = P_load*Dx*Dy # N
kp = 5e-9 # m.N-1

# cementation
# 2T   : E ( 300MPa), f_cemented (0.13), m_log (6.79), s_log (0.70)
# 2MB  : E ( 320MPa), f_cemented (0.88), m_log (7.69), s_log (0.60)
# 11BB : E ( 760MPa), f_cemented (0.98), m_log (8.01), s_log (0.88)
# 13BT : E ( 860MPa), f_cemented (1.00), m_log (8.44), s_log (0.92)
# 13MB : E (1000MPa), f_cemented (1.00), m_log (8.77), s_log (0.73)
young_modulus = 860e6 # Pa
f_cemented = 1.0 # -
m_log = 8.44 # -
s_log = 0.92 # -
tensileCohesion = 2.75e6 # Pa
shearCohesion = 6.6e6 # Pa

# Dissolution of bond
f_Sc_diss = 5e-3
dSc_dissolved = f_Sc_diss*np.exp(m_log)*1e-12
f_n_bond_stop = 0

# Dissolution of grain
f_dR_diss = 1e-6
dR_dissolved = f_dR_diss*rMean
f_rMean_stop = 0.5

# time step
factor_dt_crit = 0.6

# steady-state detection
unbalancedForce_criteria = 0.01

# Report
simulation_report_name = O.tags['d.id']+'_report.txt'
simulation_report = open(simulation_report_name, 'w')
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
O.materials.append(CohFrictMat(young=young_modulus, poisson=0.25, frictionAngle=0, density=2650, isCohesive=False, momentRotationLaw=False))
# create box and grains
O.bodies.append(aabbWalls([Vector3(0,0,0),Vector3(Dx,Dy,Dz)], thickness=0.,oversizeFactor=1))
# a list of 6 boxes Bodies enclosing the packing, in the order minX, maxX, minY, maxY, minZ, maxZ
lateral_plate = O.bodies[0]
upper_plate = O.bodies[-1]

# define grain material
O.materials.append(CohFrictMat(young=young_modulus, poisson=0.25, frictionAngle=atan(0.05), density=2650,\
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
    global iter_0
    # the rest will be run only if unbalanced is < .1 (stabilized packing)
    # Compute the ratio of mean summary force on bodies and mean force magnitude on interactions.
    if unbalancedForce() > .1:
        return
    if int(O.tags['Step ic']) < n_steps_ic :
        print('IC step '+O.tags['Step ic']+'/'+str(n_steps_ic)+' done')
        iter_0 = O.iter
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
    iter_0 = O.iter
    global tic
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac
    #report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("IC Generated : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.write('Porosity = '+str(round(porosity(),3))+'\n')
    simulation_report.write(str(n_grains)+' grains\n\n')
    simulation_report.close()
    print("IC Generated : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    # save
    #O.save('save/simu_ic.yade.bz2')
    # next time, do not call this function anymore, but the next one instead
    checker.command = 'checkUnbalanced_ir_load_ic()'
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
        upper_plate.state.pos =  (Dx/2, Dy/2, max([b.state.pos[2]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]))
    else :
        dF = Fz - F_cementation
        v_plate_max = rMean*0.0002/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to top wall
        if v_try_abs < v_plate_max :
            upper_plate.state.vel = (0, 0, np.sign(dF)*v_try_abs)
        else :
            upper_plate.state.vel = (0, 0, np.sign(dF)*v_plate_max)

#-------------------------------------------------------------------------------

def checkUnbalanced_ir_load_ic():
    addPlotData_ic()
    saveData_ic()
    # check the force applied
    if abs(O.forces.f(upper_plate.id)[2]-F_cementation)/F_cementation > 0.01:
        return
    if unbalancedForce() > unbalancedForce_criteria :
        return
    global tic
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac
    #report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Pressure applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.write('Porosity = '+str(round(porosity(),3))+'\n')
    simulation_report.write('Force applied = '+str(int(O.forces.f(upper_plate.id)[2]))+'/'+str(int(F_load))+' N (target)\n')
    simulation_report.write(str(n_grains)+' grains\n\n')
    simulation_report.close()
    print("Pressure applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
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
    checker.command = 'checkUnbalanced_ir_load_param_ic()'

#-------------------------------------------------------------------------------

def checkUnbalanced_ir_load_param_ic():
    addPlotData_ic()
    saveData_ic()
    # check the force applied
    if abs(O.forces.f(upper_plate.id)[2]-F_cementation)/F_cementation > 0.01:
        return
    if unbalancedForce() > unbalancedForce_criteria :
        return
    # characterize the ic algorithm
    global iter_0
    iter_0 = O.iter
    global tic
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac
    #report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Parameters applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.close()
    print("Parameters applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    # reset plot (IC done, simulation starts)
    plot.reset()
    # save new reference position for upper wall
    upper_plate.state.refPos = upper_plate.state.pos
    # label step
    O.tags['Current Step']='0'
    # track unbalanced
    global L_unbalanced_ite, L_k0_ite, L_confinement_ite, L_count_bond
    L_unbalanced_ite = []
    L_k0_ite = []
    L_confinement_ite = []
    L_count_bond = []
    # save
    O.save('save/'+O.tags['d.id']+'_ic.yade.bz2')
    # next time, do not call this function anymore, but the next one instead
    checker.command = 'cementation()'
    checker.iterPeriod = 10

#-------------------------------------------------------------------------------

def addPlotData_ic():
    """
    Save data in plot.
    """
    # add forces applied on wall x and z
    Fz = O.forces.f(upper_plate.id)[2]
    Fx = O.forces.f(lateral_plate.id)[0]
    # add data
    plot.addData(i=O.iter, porosity=porosity(), coordination=avgNumInteractions(), unbalanced=unbalancedForce(),\
                 Fx=Fx, Fz=Fz, conf_verified=Fz/F_load*100, vert_strain=100*(upper_plate.state.pos[2]-upper_plate.state.refPos[2])/upper_plate.state.refPos[2])

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
    L_porosity = []
    file = 'data/IC_'+O.tags['d.id']+'.txt'
    data = np.genfromtxt(file, skip_header=1)
    file_read = open(file, 'r')
    lines = file_read.readlines()
    file_read.close()
    if len(lines) >= 3:
        for i in range(len(data)):
            L_sigma_x.append(abs(data[i][0]/(Dz*Dy)))
            L_sigma_z.append(abs(data[i][1]/(Dx*Dy)))
            L_confinement.append(data[i][2])
            L_coordination.append(data[i][3])
            L_unbalanced.append(data[i][6])
            L_ite.append(data[i][4]-iter_0)
            L_porosity.append(data[i][5])
            L_vert_strain.append(data[i][7])

        # plot
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(20,10),num=1)

        ax1.plot(L_ite, L_sigma_x, label = r'$\sigma_x$')
        ax1.plot(L_ite, L_sigma_z, label = r'$\sigma_z$')
        ax1.legend()
        ax1.set_title('Stresses (Pa)')

        ax2.plot(L_ite, L_unbalanced, 'b')
        ax2.set_ylabel('Unbalanced (-)', color='b')
        ax2b = ax2.twinx()
        ax2b.plot(L_ite, L_confinement, 'r')
        #ax2b.set_ylabel('Confinement (%)', color='r')
        ax2b.set_title('Steady-state indices')

        ax3.plot(L_ite, L_unbalanced, 'b')
        #ax3.set_ylabel('Unbalanced (-)', color='b')
        ax3.set_ylim(ymin=L_unbalanced[-1]/5, ymax=L_unbalanced[-1]*5)
        ax3b = ax3.twinx()
        ax3b.plot(L_ite, L_confinement, 'r')
        ax3b.set_ylim(ymin=50, ymax=200)
        ax3b.set_ylabel('Confinement (%)', color='r')
        ax3b.set_title('Steady-state indices (focus)')

        ax4.plot(L_ite, L_vert_strain)
        ax4.set_title(r'$\epsilon_v$ (%)')

        ax5.plot(L_ite, L_porosity)
        ax5.set_title('Porosity (-)')

        ax6.plot(L_ite, L_coordination)
        ax6.set_title('Coordination number (-)')

        plt.savefig('plot/IC_'+O.tags['d.id']+'.png')

        plt.close()

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
    simulation_report.write(str(counter_bond)+" contacts cemented\n")
    simulation_report.close()
    print(str(counter_bond)+" contacts cemented")

    # next time, do not call this function anymore, but the next one instead
    checker.command = 'checkUnbalanced()'
    checker.iterPeriod = 500
    O.engines = O.engines[:-1]
    O.engines = O.engines + [PyRunner(command='controlTopWall()', iterPeriod = 1)]

#-------------------------------------------------------------------------------
#Load
#-------------------------------------------------------------------------------

def controlTopWall():
    '''
    Control the upper wall to applied a defined confinement force.

    The displacement of the wall depends on the force difference. A maximum value is defined.
    '''
    Fz = O.forces.f(upper_plate.id)[2]
    if Fz == 0:
        upper_plate.state.pos =  (Dx/2, Dy/2, max([b.state.pos[2]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]))
    else :
        dF = Fz - F_load
        v_plate_max = rMean*0.0002/O.dt
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
        k0 = abs(O.forces.f(lateral_plate.id)[0]/(upper_plate.state.pos[2]*Dy)*(Dx*Dy)/O.forces.f(upper_plate.id)[2])
    else :
        k0 = 0
    L_k0_ite.append(k0)
    L_confinement_ite.append(O.forces.f(upper_plate.id)[2]/F_load*100)
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
       (abs(O.forces.f(upper_plate.id)[2]-F_load) < 0.01*F_load):

        # save old figure
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,9),num=1)
        ax1.plot(L_unbalanced_ite)
        ax1.set_title('unbalanced force (-)')
        ax2.plot(L_k0_ite)
        ax2.set_title(r'$k_0$ (-)')
        ax3.plot(L_confinement_ite)
        ax3.set_title('confinement (%)')
        fig.savefig('plot/tracking_prev_ite.png')
        plt.close()

        # reset trackers
        L_unbalanced_ite = []
        L_k0_ite = []
        L_confinement_ite = []
        L_count_bond = []

        # compute rMean_current
        rMean_current = 0
        n_mean = 0
        for b in O.bodies :
            if isinstance(b.shape, Sphere) :
                rMean_current = rMean_current + b.shape.radius
                n_mean = n_mean + 1
        rMean_current = rMean_current / n_mean

        if counter_bond0*f_n_bond_stop < counter_bond or rMean*f_rMean_stop < rMean_current:
            dissolve()
        else :
            stopLoad()

#-------------------------------------------------------------------------------

def dissolve():
    '''
    Dissolve grains and bonds with a constant reduction.
    '''
    # save at the end
    saveData()
    O.tags['Current Step'] = str(int(O.tags['Current Step'])+1)
    # dissolution
    dissolve_bond()
    dissolve_grain()

#-------------------------------------------------------------------------------

def dissolve_bond():
    """
    Dissolve bond with a constant surface reduction.
    """
    # count the number of bond
    global counter_bond, counter_bond_broken_diss, counter_bond_broken_load
    counter_bond = 0
    counter_bond_broken = 0
    # iterate on interactions
    for i in O.interactions:
        # only grain-grain contact can be cemented
        if isinstance(O.bodies[i.id1].shape, Sphere) and isinstance(O.bodies[i.id2].shape, Sphere) :
            if not i.phys.cohesionBroken :
                counter_bond = counter_bond + 1
                # set normal and shear adhesions
                i.phys.normalAdhesion = i.phys.normalAdhesion - tensileCohesion*dSc_dissolved
                i.phys.shearAdhesion = i.phys.shearAdhesion - shearCohesion*dSc_dissolved
                if i.phys.normalAdhesion <= 0 or i.phys.shearAdhesion <=0 :
                    # bond brokes
                    counter_bond = counter_bond - 1
                    counter_bond_broken = counter_bond_broken + 1
                    i.phys.cohesionBroken = True
                    i.phys.normalAdhesion = 0
                    i.phys.shearAdhesion = 0
    # update the counter of bond dissolved during the dissolution step
    counter_bond_broken_diss = counter_bond_broken_diss + counter_bond_broken
    counter_bond_broken_load = (counter_bond0-counter_bond) - counter_bond_broken_diss

#-------------------------------------------------------------------------------

def dissolve_grain():
    """
    Dissolve grain with a constant radius reduction.
    """
    # dissolution with a multiplier factor
    for b in O.bodies :
        if isinstance(b.shape, Sphere) :
            growParticle(b.id, max(b.shape.radius-dR_dissolved, 0)/b.shape.radius)
    # recompute the time step
    O.dt = factor_dt_crit * PWaveTimeStep()

#-------------------------------------------------------------------------------

def stopLoad():
    """
    Close simulation.
    """
    # save at the converged iteration
    saveData()
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
    simulation_report.write("Simulation time : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.close()
    print("Simulation time : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    O.pause()

    # save simulation
    os.mkdir('../AcidOedo_Calcarenite_data/'+O.tags['d.id'])
    shutil.copytree('data','../AcidOedo_Calcarenite_data/'+O.tags['d.id']+'/data')
    shutil.copytree('plot','../AcidOedo_Calcarenite_data/'+O.tags['d.id']+'/plot')
    shutil.copytree('save','../AcidOedo_Calcarenite_data/'+O.tags['d.id']+'/save')
    shutil.copy('AcidOedo_Calcarenite.py','../AcidOedo_Calcarenite_data/'+O.tags['d.id']+'/AcidOedo_Calcarenite.py')
    shutil.copy(O.tags['d.id']+'_report.txt','../AcidOedo_Calcarenite_data/'+O.tags['d.id']+'/'+O.tags['d.id']+'_report.txt')

#-------------------------------------------------------------------------------

def addPlotData():
    """
    Save data in plot.
    """
    # add forces applied on wall x and z
    Fz = O.forces.f(upper_plate.id)[2]
    Fx = O.forces.f(lateral_plate.id)[0]
    # compute the k0 = sigma_x/sigma_z, = 0 if no sigma_z
    if Fz != 0:
        k0 = abs(Fx/(upper_plate.state.pos[2]*Dy)*(Dx*Dy)/Fz)
    else :
        k0 = 0
    # compute mass of the sample
    Mass_total = 0
    for b in O.bodies :
        if isinstance(b.shape, Sphere) :
            Mass_total = Mass_total + b.state.mass
    # add data
    plot.addData(i=O.iter, porosity=porosity(), coordination=avgNumInteractions(), unbalanced=unbalancedForce(), \
                counter_bond=counter_bond, counter_bond_broken_diss=counter_bond_broken_diss, counter_bond_broken_load=counter_bond_broken_load,\
                 Fx=Fx, Fz=Fz, Z_plate=upper_plate.state.pos[2], conf_verified=Fz/F_load*100, k0=k0, Mass_total=Mass_total,\
                 w=upper_plate.state.pos[2]-upper_plate.state.refPos[2], vert_strain=100*(upper_plate.state.pos[2]-upper_plate.state.refPos[2])/upper_plate.state.refPos[2])

#-------------------------------------------------------------------------------

def saveData():
    """
    Save data in .txt file during the steps.
    """
    addPlotData()
    plot.saveDataTxt('data/'+O.tags['d.id']+'.txt')
    # post-proccess
    L_k0 = []
    L_confinement = []
    L_unbalanced = []
    L_coordination = []
    L_vert_strain = []
    L_porosity = []
    L_mass = []
    L_mass_diss = []
    L_radius_diss = []
    L_ite  = []
    L_counter_bond = []
    L_counter_bond_broken_diss = []
    L_counter_bond_broken_load = []
    file = 'data/'+O.tags['d.id']+'.txt'
    data = np.genfromtxt(file, skip_header=1)
    file_read = open(file, 'r')
    lines = file_read.readlines()
    file_read.close()
    if len(lines) >= 3:
        for i in range(len(data)):
            L_mass.append(data[i][2])
            L_confinement.append(data[i][4])
            L_coordination.append(data[i][5])
            L_counter_bond.append(data[i][6])
            L_counter_bond_broken_diss.append(data[i][7])
            L_counter_bond_broken_load.append(data[i][8])
            L_ite.append(data[i][9]-iter_0)
            L_k0.append(data[i][10])
            L_porosity.append(data[i][11])
            L_unbalanced.append(data[i][12])
            L_vert_strain.append(data[i][13])
            L_mass_diss.append(100*(L_mass[0]-L_mass[-1])/L_mass[0])
            L_radius_diss.append(100*(1-(1-L_mass_diss[-1]/100)**(1/3)))

        # plot
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(16,9),num=1)

        ax1.plot(L_k0)
        ax1.set_title(r'$k_0$ (-) - step (-)')

        ax2.plot(L_counter_bond, label='intact')
        ax2.plot(L_counter_bond_broken_diss, label='broken during dissolution')
        ax2.plot(L_counter_bond_broken_load, label='broken during loading')
        ax2.set_title('Number of bond (-) - step (-)')
        ax2.legend()

        ax3.plot(L_mass_diss, color='r')
        ax3.set_ylabel('Mass (%)', color='r')
        ax3b = ax3.twinx()
        ax3b.plot(L_radius_diss, color = 'b')
        ax3b.set_ylabel('Radius (%)', color = 'b')
        ax3.set_title('Dissolution (-) - step (-)')

        ax4.plot(L_vert_strain)
        ax4.set_title(r'$\epsilon_v$ (%) - step (-)')

        ax5.plot(L_porosity)
        ax5.set_title('Porosity (-) - step (-)')

        ax6.plot(L_coordination)
        ax6.set_title('Coordination (-) - step (-)')

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
