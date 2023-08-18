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

# batch
readParamsFromTable(freq_diss=50, f_dR_diss=1e-6, n_step_ic=100, damping=0.001, kp=1e-9)
from yade.params import table

#-------------------------------------------------------------------------------
#User
#-------------------------------------------------------------------------------

# PSD
n_grains = 3000
L_r = []

# Particles
rMean = 0.00125  # m
rRelFuzz = .2

# Box
Dz_on_Dx = 1 # ratio Dz / Dxy
Dz = 0.036 # m
Dx = Dz/Dz_on_Dx
Dy = Dx

# IC
n_steps_ic = table.n_step_ic

# Top wall
P_load = 1e5 # Pa
F_load = P_load*Dx*Dy # N
kp = table.kp # m.N-1

# Dissolution
f_diss = 0.25 # part dissolved
f_dR_diss = table.f_dR_diss
dR_dissolved = f_dR_diss*rMean
freq_diss = table.freq_diss
step_max = freq_diss*rMean/dR_dissolved*0.1

# time step
factor_dt_crit = 0.6

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
O.materials.append(FrictMat(young = 7.54e9, poisson = 0.3, frictionAngle = 0, density = 2500))
# create box and grains
O.bodies.append(wall(0 , axis=0, sense=0)) # -x
lateral_plate = O.bodies[-1] # the last body is a lateral body
O.bodies.append(wall(Dx, axis=0, sense=0)) # +x
O.bodies.append(wall(0 , axis=1, sense=0)) # -y
O.bodies.append(wall(Dy, axis=1, sense=0)) # +y
O.bodies.append(wall(0 , axis=2, sense=0)) # -z
O.bodies.append(wall(Dz, axis=2, sense=0)) # +z
upper_plate = O.bodies[-1]  # the last body is the upper plate

# define grain material
O.materials.append(FrictMat(young = 7.54e9, poisson = 0.3, frictionAngle = atan(0.05), density = 2500))
L_id_diss = []
for i in range(n_grains):
    radius = random.uniform(rMean*(1-rRelFuzz),rMean*(1+rRelFuzz))
    center_x = random.uniform(0+radius/n_steps_ic, Dx-radius/n_steps_ic)
    center_y = random.uniform(0+radius/n_steps_ic, Dy-radius/n_steps_ic)
    center_z = random.uniform(0+radius/n_steps_ic, Dz-radius/n_steps_ic)
    O.bodies.append(sphere(center=[center_x, center_y, center_z], radius=radius/n_steps_ic))
    O.bodies[-1].state.blockedDOFs = 'XYZ'
    # can use b.state.blockedDOFs = 'xyzXYZ' to block translation of rotation of a body
    L_r.append(radius)
    # determine if the grain is dissolvable
    if random.uniform(0,1) < f_diss :
        L_id_diss.append(O.bodies[-1].id)
O.tags['Step ic'] = '1'

# yade algorithm
O.engines = [
        ForceResetter(),
        # sphere, wall
        InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Wall_Aabb()]),
        InteractionLoop(
                # need to handle sphere+sphere and sphere+wall
                # Ig : compute contact point. Ig2_Sphere (3DOF) or Ig2_Sphere6D (6DOF)
                # Ip : compute parameters needed
                # Law : compute contact law with parameters from Ip
                [Ig2_Sphere_Sphere_ScGeom6D(), Ig2_Wall_Sphere_ScGeom()],
                [Ip2_FrictMat_FrictMat_MindlinPhys()],
                [Law2_ScGeom_MindlinPhys_Mindlin()]
        ),
        NewtonIntegrator(gravity=(0, 0, 0), damping=table.damping, label = 'Newton'),
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
    #binsSizes, binsProc, binsSumCum = psd(bins=10)
    #plt.figure(1, figsize=(16,9))
    #plt.plot(binsSizes, binsProc)
    #plt.title('Particle Size Distribution')
    #plt.savefig('plot/PSD_0.png')
    #plt.close()
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
    O.engines = O.engines + [PyRunner(command='controlTopWall()', iterPeriod = 1)]
    # switch on friction between particle
    O.materials[-1].frictionAngle = atan(0.5)
    # for existing contacts, set contact friction directly
    for i in O.interactions :
        i.phys.tangensOfFrictionAngle = tan(atan(0.5))
    # decrease the damping
    Newton.damping = 0.001
    # switch on the gravity
    Newton.gravity = [0, 0, -9.81]

#-------------------------------------------------------------------------------

def checkUnbalanced_ir_load_ic():
    global kp
    addPlotData_ic()
    saveData_ic()
    # check the force applied
    if abs(O.forces.f(upper_plate.id)[2]-F_load)/F_load > 0.01:
        return
    if unbalancedForce() > 0.001:
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
    simulation_report.write("Pressure applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.write('Porosity = '+str(round(porosity(),3))+'\n')
    simulation_report.write('Force applied = '+str(int(O.forces.f(upper_plate.id)[2]))+'/'+str(int(F_load))+' N (target)\n')
    simulation_report.write(str(n_grains)+' grains\n\n')
    simulation_report.close()
    print("Pressure applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    # save
    O.save('save/'+O.tags['d.id']+'_ic.yade.bz2')
    # next time, do not call this function anymore, but the next one instead
    checker.command = 'checkUnbalanced()'
    checker.iterPeriod = freq_diss
    # start plotting the data now, it was not interesting before
    O.engines = O.engines + [PyRunner(command='addPlotData()', iterPeriod = int(freq_diss/10), label='plotter')]
    plot.reset()
    # switch off damping
    Newton.damping = 0
    # label step
    O.tags['Current Step']='0'
    # save new reference position for upper wall
    upper_plate.state.refPos = upper_plate.state.pos

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

        ax4.plot(L_ite, L_porosity)
        ax4.set_title('Porosity (-)')

        ax5.plot(L_ite, L_coordination)
        ax5.set_title('Coordination number (-)')

        ax6.plot(L_ite, L_vert_strain)
        ax6.set_title(r'$\epsilon_v$ (%)')

        plt.savefig('plot/IC_'+O.tags['d.id']+'.png')

        plt.close()

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
        upper_plate.state.pos =  (0, 0, max([b.state.pos[2]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]))
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

def checkUnbalanced():
    """
    Look for the steady state during the loading phase.
    """
    # next time, do not call this function anymore, but the next one instead
    if int(O.tags['Current Step']) < step_max:
        dissolveGrains()
    else :
        stopLoad()

#-------------------------------------------------------------------------------

def dissolveGrains():
    """
    Dissolve grain with a constant radius reduction.
    """
    # save at the end
    saveData()
    plt.close()
    O.tags['Current Step'] = str(int(O.tags['Current Step'])+1)
    # dissolution with a multiplier factor
    for b in O.bodies :
        if isinstance(b.shape, Sphere) and b.id in L_id_diss:
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
    os.mkdir('../data_Investigation_Cha/'+O.tags['d.id'])
    shutil.copytree('data','../data_Investigation_Cha/'+O.tags['d.id']+'/data')
    shutil.copytree('plot','../data_Investigation_Cha/'+O.tags['d.id']+'/plot')
    shutil.copytree('save','../data_Investigation_Cha/'+O.tags['d.id']+'/save')
    shutil.copy('like_cha_santamarina.py','../data_Investigation_Cha/'+O.tags['d.id']+'/like_cha_santamarina.py')
    shutil.copy(O.tags['d.id']+'_report.txt','../data_Investigation_Cha/'+O.tags['d.id']+'/'+O.tags['d.id']+'_report.txt')

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
        if isinstance(b.shape, Sphere) and b.id in L_id_diss:
            Mass_total = Mass_total + b.state.mass
    # add data
    plot.addData(i=O.iter, porosity=porosity(), coordination=avgNumInteractions(), unbalanced=unbalancedForce(),\
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
    L_ite  = []
    file = 'data/'+O.tags['d.id']+'.txt'
    data = np.genfromtxt(file, skip_header=1)
    file_read = open(file, 'r')
    lines = file_read.readlines()
    file_read.close()
    if len(lines) >= 3:
        for i in range(len(data)):
            L_k0.append(data[i][7])
            L_confinement.append(data[i][4])
            L_unbalanced.append(data[i][9])
            L_coordination.append(data[i][5])
            L_vert_strain.append(data[i][10])
            L_porosity.append(data[i][8])
            L_mass.append(data[i][2])
            L_mass_diss.append(100*(L_mass[0]-L_mass[-1])/L_mass[0])
            L_ite.append(data[i][6]-iter_0)

        # plot
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(16,9),num=1)

        ax1.plot(L_mass_diss, L_k0)
        ax1.set_title(r'$k_0$ (-) - mass dissolved (%)')

        ax2.plot(L_mass_diss, L_unbalanced)
        ax2.set_title('Unbalanced (-) - mass dissolved (%)')

        ax3.plot(L_mass_diss, L_confinement)
        ax3.set_title('Confinement (%) - mass dissolved (%)')

        ax4.plot(L_mass_diss, L_vert_strain)
        ax4.set_title(r'$\epsilon_v$ (%) - mass dissolved (%)')

        ax5.plot(L_mass_diss, L_porosity)
        ax5.set_title('Porosity (-) - mass dissolved (%)')

        ax6.plot(L_mass_diss, L_coordination)
        ax6.set_title('Coordination (-) - mass dissolved (%)')

        plt.savefig('plot/'+O.tags['d.id']+'.png')
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
        print('\tMean R =',round(1-int(O.tags['Current Step'])*dR_dissolved/rMean,2),'initial mean radius')
        print('\tMass =',round((1-int(O.tags['Current Step'])*dR_dissolved/rMean)**3,2),'initial mass')
        print('Sample description :')
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
