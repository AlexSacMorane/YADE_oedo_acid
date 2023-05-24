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
import os
import shutil

#-------------------------------------------------------------------------------
#User
#-------------------------------------------------------------------------------

# PSD
n_grains = 9000
L_r = []
size_ratio_coarse_fine = 3

# Coarse particles
rMean_1 = 0.001  # m
rRelFuzz_1 = .2

# Fines
rMean_2 = rMean_1/size_ratio_coarse_fine  # m
rRelFuzz_2 = .2

# Coarse-Fine distribution
number_ratio_coarse_total = 0.4
rMean = number_ratio_coarse_total*rMean_1 + (1-number_ratio_coarse_total)*rMean_2

# Box
n_g_on_z = 20 # height / (2*Rmean)
Dz_on_Dx = 0.7 # ratio Dz / Dxy
Dz = n_g_on_z * 2*rMean
Dx = Dz/Dz_on_Dx
Dy = Dx

# IC
n_steps_ic = 30
counter_checked_target = 2
counter_checked = 0
expansion_load = 3*1e-4

# Top wall
P_load = 1e5 # Pa
F_load = P_load*Dx*Dy # N
kp = 1e-10 # m.N-1

# Dissolution
dR_dissolved = 0.003*rMean
dR_dissolved_focus = dR_dissolved/3
n_step_focus = 15
# step_max-1 dissolution steps as first step is with initial size
step_max = 30

# time step
factor_dt_crit = 0.6

# Report
simulation_report_name = 'simulation_report.txt'
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
os.mkdir('plot/Contact_Orientation')
os.mkdir('plot/Step')
if Path('data').exists():
    shutil.rmtree('data')
os.mkdir('data')
if Path('vtk').exists():
    shutil.rmtree('vtk')
os.mkdir('vtk')
if Path('save').exists():
    shutil.rmtree('save')
os.mkdir('save')
vtkExporter = export.VTKExporter('vtk/ic')

# define wall material (no friction)
O.materials.append(FrictMat(young = 70e9, poisson = 0.3, frictionAngle = 0, density = 2500))
# create box and grains
O.bodies.append(geom.facetBox((Dx/2, Dy/2, Dz/2), (Dx/2, Dy/2, Dz/2), wallMask=30))
O.bodies.append(wall(0, axis=0, sense=0))
lateral_plate = O.bodies[-1]
O.bodies.append(wall(Dz, axis=2, sense=-1))
upper_plate = O.bodies[-1]  # the last particles is the plate
# define grain material
O.materials.append(FrictMat(young = 70e9, poisson = 0.3, frictionAngle = 0.5, density = 2500))
for i in range(n_grains):
    # coarse
    if random.uniform(0,1) < number_ratio_coarse_total :
        radius = random.uniform(rMean_1*(1-rRelFuzz_1),rMean_1*(1+rRelFuzz_1))
    # fine
    else :
        radius = random.uniform(rMean_2*(1-rRelFuzz_2),rMean_2*(1+rRelFuzz_2))
    center_x = random.uniform(0+radius/n_steps_ic, Dx-radius/n_steps_ic)
    center_y = random.uniform(0+radius/n_steps_ic, Dy-radius/n_steps_ic)
    center_z = random.uniform(0+radius/n_steps_ic, Dz-radius/n_steps_ic)
    O.bodies.append(sphere(center=[center_x, center_y, center_z], radius=radius/n_steps_ic))
    L_r.append(radius)
O.tags['Step ic'] = '1'

# yade algorithm
O.engines = [
        ForceResetter(),
        # sphere, facet, wall
        InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Facet_Aabb(), Bo1_Wall_Aabb()]),
        InteractionLoop(
                # need to handle sphere+sphere, sphere+facet, sphere+wall
                # Ig : compute contact point. Ig2_Sphere/Facet_Sphere_ScGeom (3DOF) or Ig2_Sphere/Facet_Sphere_ScGeom6D (6DOF)
                # Ip : compute parameters needed
                # Law : compute contact law with parameters from Ip
                [Ig2_Sphere_Sphere_ScGeom6D(), Ig2_Facet_Sphere_ScGeom6D(), Ig2_Wall_Sphere_ScGeom()],
                [Ip2_FrictMat_FrictMat_MindlinPhys()],
                [Law2_ScGeom_MindlinPhys_Mindlin()]
        ),
        NewtonIntegrator(gravity=(0, 0, -9.81), damping=0.3, label = 'Newton'),
        PyRunner(command='checkUnbalanced_ir_ic()', iterPeriod = 200, label='checker')
]
# time step
O.dt = factor_dt_crit * PWaveTimeStep()
# start simulation
O.run()

#-------------------------------------------------------------------------------

def checkUnbalanced_ir_ic():
    '''
    Increase particle radius until a steady-state is found.
    '''
    global iter_0
	# at the very start, unbalanced force can be low as there is only few contacts, but it does not mean the packing is stable
    if O.iter < iter_0 + 1000:
        return
	# the rest will be run only if unbalanced is < .1 (stabilized packing)
    # Compute the ratio of mean summary force on bodies and mean force magnitude on interactions.
    if unbalancedForce() > .05:
        return
    if int(O.tags['Step ic']) < n_steps_ic :
        print('IC step '+O.tags['Step ic']+'/'+str(n_steps_ic)+' done')
        # export vtk file
        vtkExporter.exportSpheres()
        iter_0 = O.iter
        O.tags['Step ic'] = str(int(O.tags['Step ic'])+1)
        i_L_r = 0
        for b in O.bodies :
            if isinstance(b.shape, Sphere):
                growParticle(b.id, int(O.tags['Step ic'])/n_steps_ic*L_r[i_L_r]/b.shape.radius)
                i_L_r = i_L_r + 1
        O.dt = factor_dt_crit * PWaveTimeStep()
        return
    # export vtk file
    vtkExporter.exportSpheres()
    # plot the psd
    binsSizes, binsProc, binsSumCum = psd(bins=10)
    plt.figure(1, figsize=(16,9))
    plt.plot(binsSizes, binsProc)
    plt.title('Particle Size Distribution')
    plt.savefig('plot/PSD_0.png')
    plt.close()
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
    simulation_report.write('Porosity = '+str(round(porosity(),2))+'/'+str(porosity_target)+' (target)\n')
    simulation_report.write(str(n_grains)+' grains\n\n')
    simulation_report.close()
    print("IC Generated : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    # save
    O.save('save/simu_ic.yade.bz2')
    # next time, do not call this function anymore, but the next one instead
    checker.command = 'checkUnbalanced_ir_load_ic()'
    checker.iterPeriod = 700

#-------------------------------------------------------------------------------

def checkUnbalanced_ir_load_ic():
    # export vtk file
    global vtkExporter
    global counter_checked
    addPlotData_ic()
    saveData_ic()
    # check the force applied
    if O.forces.f(upper_plate.id)[2] < F_load:
        vtkExporter.exportSpheres()
        growParticles(1+expansion_load)
        O.dt = .5 * PWaveTimeStep()
        return
    elif 100*F_load < O.forces.f(upper_plate.id)[2]:
        vtkExporter.exportSpheres()
        growParticles(1-expansion_load)
        O.dt = .5 * PWaveTimeStep()
        return
    else :
        counter_checked = counter_checked + 1
    if counter_checked <= counter_checked_target :
        return
    if unbalancedForce() > 0.01:
        return
    # plot the psd
    binsSizes, binsProc, binsSumCum = psd(bins=10)
    plt.figure(1, figsize=(16,9))
    plt.plot(binsSizes, binsProc)
    plt.title('Particle Size Distribution')
    plt.savefig('plot/PSD_1.png')
    plt.close()
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
    simulation_report.write('Porosity = '+str(porosity())+'/'+str(porosity_target)+' (target)\n')
    simulation_report.write('Force applied = '+str(int(O.forces.f(upper_plate.id)[2]))+'/'+str(F_load)+' N (target)\n')
    simulation_report.write(str(n_grains)+' grains\n\n')
    simulation_report.close()
    print("Pressure applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    # save
    O.save('save/simu_ic_load.yade.bz2')
    # next time, do not call this function anymore, but the next one instead
    checker.command = 'checkUnbalanced()'
    checker.iterPeriod = 200
	# control top wall
    O.engines = O.engines + [PyRunner(command='controlTopWall()', iterPeriod = 1)]
    # start plotting the data now, it was not interesting before
    O.engines = O.engines + [PyRunner(command='addPlotData()', iterPeriod = 250, label='plotter')]
    O.engines = O.engines + [PyRunner(command='saveData()', iterPeriod = 500, label='saver')]
    plot.reset()
    # label step
    O.tags['Current Step']='1'
    # change the vtk exporter
    vtkExporter = export.VTKExporter('vtk/sample')

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
                 Fx=Fx, Fz=Fz, conf_verified=Fz/F_load*100)

#-------------------------------------------------------------------------------

def saveData_ic():
    """
    Save data in .txt file during the ic.
    """
    plot.saveDataTxt('data/IC.txt')
    # post-proccess
    L_sigma_x = []
    L_sigma_z = []
    L_confinement = []
    L_coordination = []
    L_unbalanced = []
    L_ite  = []
    L_porosity = []
    file = 'data/IC.txt'
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

        # plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,9),num=1)

        ax1.plot(L_ite, L_sigma_x, label = r'$\sigma_x$')
        ax1.plot(L_ite, L_sigma_z, label = r'$\sigma_z$')
        ax1.legend()
        ax1.set_title('Stresses (Pa)')

        ax2.plot(L_ite, L_unbalanced, 'b')
        ax2.set_ylabel('Unbalanced (-)', color='b')
        ax2b = ax2.twinx()
        ax2b.plot(L_ite, L_confinement, 'r')
        ax2b.set_ylabel('Confinement (%)', color='r')
        ax2b.set_ylim(ymin=0, ymax=200)
        ax2b.set_title('Steady-state indices')

        ax3.plot(L_ite, L_porosity)
        ax3.set_title('Porosity (-)')

        ax4.plot(L_ite, L_coordination)
        ax4.set_title('Coordination number (-)')

        plt.savefig('plot/IC.png')
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
        v_plate_max = rMean*0.0001/O.dt
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
    global iter_0
    # infinite iterations not allowed
    if O.iter > iter_0+3000000:
        #report
        simulation_report = open(simulation_report_name, 'a')
        simulation_report.write('\nSteady state not found')
        simulation_report.close()
        print('Steady state not found')
        checker.command = 'stopLoad()'
	# at the very start, unbalanced force can be low as there is only few contacts, but it does not mean the packing is stable
    if O.iter < iter_0+1000:
        return
    # check the confinement force
    Fz = O.forces.f(upper_plate.id)[2]
    if abs(Fz-F_load) > 0.02*F_load:
        return
	# check unbalanced
    if unbalancedForce() > .005:
        return
    # characterize the dem step
    iter_0 = O.iter
    global tic
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    #report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("End of step "+O.tags['Current Step']+" : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.close()
    print("End of step "+O.tags['Current Step']+" : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    tic = tac
	# next time, do not call this function anymore, but the next one instead
    if int(O.tags['Current Step']) < step_max:
        checker.command = 'dissolveGrains()'
    else :
        checker.command = 'stopLoad()'

#-------------------------------------------------------------------------------

def dissolveGrains():
    """
    Dissolve grain with a constant radius reduction.
    """
    # export vtk file
    vtkExporter.exportSpheres()
    # save at the end
    plot.saveDataTxt('data/Step_' + O.tags['Current Step'] + '.txt')
    # post-proccess
    L_sigma_x = []
    L_sigma_z = []
    L_confinement = []
    L_coordination = []
    L_unbalanced = []
    L_vert_strain = []
    file = 'data/Step_'+O.tags['Current Step']+'.txt'
    data = np.genfromtxt(file, skip_header=1)
    for i in range(len(data)):
        L_sigma_x.append(abs(data[i][0]/(data[i][3]*Dy)))
        L_sigma_z.append(abs(data[i][1]/(Dx*Dy)))
        L_confinement.append(data[i][4])
        L_coordination.append(data[i][5])
        L_unbalanced.append(data[i][9])
        L_vert_strain.append(data[i][10])

    # plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,9),num=1)

    ax1.plot(L_sigma_x, label = r'$\sigma_x$')
    ax1.plot(L_sigma_z, label = r'$\sigma_z$')
    ax1.legend()

    ax2.plot(L_unbalanced, 'b')
    ax2.set_ylabel('Unbalanced', color='b')
    ax2b = ax2.twinx()
    ax2b.plot(L_confinement, 'r')
    ax2b.set_ylabel('Confinement', color='r')

    ax3.plot(L_vert_strain)
    ax3.set_title(r'$\epsilon_v$')

    ax4.plot(L_coordination)
    ax4.set_title('Coordination number')

    plt.savefig('plot/Step/Step_'+O.tags['Current Step']+'.png')
    plt.close()

    # contact Directions
    plotDirections(noShow=True, sphSph=True).savefig('plot/Contact_Orientation/Contact_Orientation_step_'+O.tags['Current Step']+'.png')

    # next step
    plot.reset()
    O.tags['Current Step'] = str(int(O.tags['Current Step'])+1)
    # dissolution with a multiplier factor
    for b in O.bodies :
        if isinstance(b.shape, Sphere):
            if int(O.tags['Current Step']) <= n_step_focus :
                growParticle(b.id, max(b.shape.radius-dR_dissolved_focus, 0)/b.shape.radius)
            else :
                growParticle(b.id, max(b.shape.radius-dR_dissolved, 0)/b.shape.radius)
    # recompute the time step
    O.dt = factor_dt_crit * PWaveTimeStep()
    # reload the sample
    checker.command = 'checkUnbalanced()'

#-------------------------------------------------------------------------------

def stopLoad():
    """
    Close simulation.
    """
    # export the vtk file
    vtkExporter.exportSpheres()
    # save at the converged iteration
    plot.saveDataTxt('data/Step_' + O.tags['Current Step'] + '.txt')
    # characterize the last DEM step and the simulation
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    #report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("End of step "+O.tags['Current Step']+" : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.close()
    print("End of step "+O.tags['Current Step']+" : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    hours = (tac-tic_0)//(60*60)
    minutes = (tac-tic_0 -hours*60*60)//(60)
    seconds = int(tac-tic_0 -hours*60*60 -minutes*60)
    #report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Simulation time : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.close()
    print("Simulation time : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    O.pause()
    # post proccess
    L_k0 = []
    L_mass = []
    L_mass_diss = []
    L_coordination = []
    L_porosity = []
    L_vert_strain = []
    for ite in range(1,step_max+1):
        file = 'data/Step_'+str(ite)+'.txt'
        data = np.genfromtxt(file, skip_header=1)
        L_k0.append(data[-1][7])
        L_mass.append(data[-1][2])
        L_mass_diss.append(100*(L_mass[0]-L_mass[-1])/L_mass[0])
        L_coordination.append(data[-1][5])
        L_porosity.append(data[-1][8])
        L_vert_strain.append(data[-1][10])
    # plot
    plt.figure(1, figsize = (16,9))

    plt.subplot(221)
    plt.plot(L_mass_diss, L_k0)
    plt.title(r'% mass dissolved - $k_0$')

    plt.subplot(222)
    plt.plot(L_mass_diss, L_coordination)
    plt.title('% mass dissolved - coordination number')

    plt.subplot(223)
    plt.plot(L_mass_diss, L_porosity)
    plt.title('% mass dissolved - porosity')

    plt.subplot(224)
    plt.plot(L_mass_diss, L_vert_strain)
    plt.title(r'% mass dissolved - $\epsilon_v$')

    plt.savefig('plot/Result.png')
    plt.close()

    #save
    outfile = open('result.data', 'wb')
    dict_save = {
    'L_k0' : L_k0,
    'L_mass' : L_mass,
    'L_mass_diss' : L_mass_diss,
    'L_coordination' : L_coordination,
    'L_porosity' : L_porosity,
    'L_vert_strain' : L_vert_strain
    }
    pickle.dump(dict_save,outfile)
    outfile.close()

    #copy and paste into save folder
    name_actual_folder = '../oedo_acid'
    shutil.copytree(name_actual_folder, '../data_oedo_acid/'+O.tags['isoTime'])

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
        if isinstance(b.shape, Sphere):
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
    plot.saveDataTxt('data/Step_' + O.tags['Current Step'] + '.txt')
    # post-proccess
    L_sigma_x = []
    L_sigma_z = []
    L_confinement = []
    L_coordination = []
    L_unbalanced = []
    L_vert_strain = []
    L_ite  = []
    file = 'data/Step_'+O.tags['Current Step']+'.txt'
    data = np.genfromtxt(file, skip_header=1)
    file_read = open(file, 'r')
    lines = file_read.readlines()
    file_read.close()
    if len(lines) >= 3:
        for i in range(len(data)):
            L_sigma_x.append(abs(data[i][0]/(data[i][3]*Dy)))
            L_sigma_z.append(abs(data[i][1]/(Dx*Dy)))
            L_confinement.append(data[i][4])
            L_coordination.append(data[i][5])
            L_unbalanced.append(data[i][9])
            L_vert_strain.append(data[i][10])
            L_ite.append(data[i][6]-iter_0)

        # plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,9),num=1)

        ax1.plot(L_ite, L_sigma_x, label = r'$\sigma_x$')
        ax1.plot(L_ite, L_sigma_z, label = r'$\sigma_z$')
        ax1.legend()
        ax1.set_title('Stresses (Pa)')

        ax2.plot(L_ite, L_unbalanced, 'b')
        ax2.set_ylabel('Unbalanced (-)', color='b')
        ax2b = ax2.twinx()
        ax2b.plot(L_ite, L_confinement, 'r')
        ax2b.set_ylabel('Confinement (%)', color='r')
        ax2b.set_ylim(ymin=0, ymax=200)
        ax2b.set_title('Steady-state indices')

        ax3.plot(L_ite, L_vert_strain)
        ax3.set_title(r'$\epsilon_v$ (%)')

        ax4.plot(L_ite, L_coordination)
        ax4.set_title('Coordination number (-)')

        plt.savefig('plot/Step/Step_'+O.tags['Current Step']+'.png')
        plt.close()
