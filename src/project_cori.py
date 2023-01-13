from cgitb import reset
from flow import FlowProject
import signac
import flow
import matplotlib.pyplot as plt
import mbuild as mb
import mdtraj as md
import numpy as np
from foyer import Forcefield
import ilforcefields.ilforcefields as ilff
import os
from get_sol_il_xml import GetSolv, GetIL, Get_ff_path
import environment_for_nersc
# import environment_for_rahman
from scipy import stats
import scipy.constants as constants
import itertools as it
from copy import deepcopy
import sympy as sym
from mtools.gromacs.gromacs import make_comtrj
from mtools.gromacs.gromacs import unwrap_trj
from mtools.post_process import calc_msd
from mtools.post_process import compute_cn
from itertools import combinations_with_replacement 
# from calc_transport import calc_ne_conductivity, calc_eh_conductivity
import MDAnalysis as mda
import scipy.integrate as sci_integrate
import unyt as u_1
import model_builder
from mbuild.formats import lammpsdata
# import lammpsdata
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def workspace_command(cmd):
    """Simple command to always go to the workspace directory"""
    return " && ".join(
        [
            "cd {job.ws}",
            cmd if not isinstance(cmd, list) else " && ".join(cmd),
            "cd ..",
        ]
    )
def _run_overall(trj, mol):
    D, MSD, x_fit, y_fit = calc_msd(trj)
    return D, MSD

def _save_overall(job, mol, trj, MSD):
        np.savetxt(os.path.join(job.workspace(), 'msd-{}-overall.txt'.format(mol)),
                        np.transpose(np.vstack([trj.time, MSD])),
                                header='# Time (ps)\tMSD (nm^2)')
        fig, ax = plt.subplots()
        ax.plot(trj.time, MSD)
        ax.set_xlabel('Simulation time (ps)')
        ax.set_ylabel('MSD (nm^2)')
        fig.savefig(os.path.join(job.workspace(),
                    'msd-{}-overall.pdf'.format(mol)))
        
def _save_overall_solo(job, mol, trj, MSD):
        np.savetxt(os.path.join(job.workspace(), 'msd-solo-{}-overall.txt'.format(mol)),
                        np.transpose(np.vstack([trj.time, MSD])),
                                header='# Time (ps)\tMSD (nm^2)')
        fig, ax = plt.subplots()
        ax.plot(trj.time, MSD)
        ax.set_xlabel('Simulation time (ps)')
        ax.set_ylabel('MSD (nm^2)')
        fig.savefig(os.path.join(job.workspace(),
                    'msd-solo-{}-overall.pdf'.format(mol)))

def _run_multiple(trj):
    D_pop = list()
    num_frame = trj.n_frames
    chunk = 5000
    for start_frame in np.linspace(0, num_frame - chunk, num = 200, dtype=np.int):
        end_frame = start_frame + chunk
        sliced_trj = trj[start_frame:end_frame]
        D_pop.append(calc_msd(sliced_trj)[0])
        
#         if end_frame < num_frame:
#             sliced_trj = trj[start_frame:end_frame]
# #             print('\t\t\t...frame {} to {}'.format(start_frame, end_frame))
#             try:
#                 D_pop.append(calc_msd(sliced_trj)[0])
#             except TypeError:
#                 import pdb
#                 pdb.set_trace()
#         else:
#             continue
    D_avg = np.mean(D_pop)
    D_std = np.std(D_pop)
    return D_avg, D_std
init_file = "system.gro"
init_solo_file = "system_solo.gro"
em_file = "em.gro"
em_solo_file = "em_solo.gro"
nvt1_file = "nvt1.gro"
nvt_solo_file = "nvt_solo.gro"
nvt2_file = "nvt2.gro"
nvt3_file = "nvt3.gro"
npt_solo_file = "npt_solo.gro"
equil_file = "equil.gro"
equil_solo_file = "equil_solo.gro"
sample_file = "sample.gro"
sample1_file = "sample1.gro"
sample_solo_file = "sample_solo.gro"
sample_velo_file = "sample_velo.gro"
total_simulation_solo_file = "total_simulation_solo.txt"
unwrapped_file = 'com.gro'
unwrapped_solo_file = 'sample_solo_unwrapped.xtc'
msd_file = 'diffusivity_and_msd_done.txt_1'
msd_solo_file = 'diffusivity_solo_done.txt'
rdf_file = 'rdf_done.txt'
rdf_solo_file = 'rdf_solo_done_1.txt'
conductivity_file = 'conductivity_done.txt'
conductivity_solo_file = 'conductivity_solo_done.txt'
cn_file = 'cn_done_5.txt'
cn_solo_file = 'cn_solo_done_3.txt'
velocity_file = 'velocity_done.txt'
gk_file = 'gk_done.txt'
density_gmx_file = 'sample_density.xvg'
auto_calibrate_file = 'auto_calibration.txt'
for_lammps_data = "sample.data"
conp_file = "restart.final"
lammps_init_file =  "sample.data"
new_conp_file = 'new_conp.lammpstrj'
conpxtc_file = "xtc.txt"

class Project(FlowProject):
    pass

@Project.label
def initialized(job):
    return job.isfile(init_file)

@Project.label
def initialize_soloed(job):
    return job.isfile(init_solo_file)

@Project.label
def minimized(job):
    return job.isfile(em_file)

@Project.label
def minimize_soloed(job):
    return job.isfile(em_solo_file)

@Project.label
def nvt1_done(job):
    return job.isfile(nvt1_file)

@Project.label
def nvt_solo_done(job):
    return job.isfile(nvt_solo_file)

@Project.label
def nvt2_done(job):
    return job.isfile(nvt2_file)

@Project.label
def npt_solo_done(job):
    return job.isfile(npt_solo_file)

# @Project.label
# def equil_done(job):
#     return job.isfile(equil_file)

# @Project.label
# def equil_solo_done(job):
#     return job.isfile(equil_solo_file)

@Project.label
def sampled(job):
    return job.isfile(sample_file)


@Project.label
def sample1d(job):
    return job.isfile(sample1_file)

@Project.label
def totaled(job):
    return job.isfile(sample_file)

@Project.label
def autocalibrated(job):
    return job.isfile(auto_calibrate_file)


# @Project.label
# def for_lammpsed(job):
#     return job.isfile(for_lammps_data)


@Project.label
def lmp_initecopyed(job):
    return job.isfile(lammps_init_file)

@Project.label
def run_cpmed(job):
    return job.isfile(conp_file)

@Project.label
def prepared(job):
    return job.isfile(unwrapped_file)

@Project.label
def density_gmxed(job):
    return job.isfile(density_gmx_file)

@Project.label
def msd_done(job):
    return job.isfile(msd_file)

@Project.label
def msd_solo_done(job):
    return job.isfile(msd_solo_file)

@Project.label
def rdf_solo_done(job):
    return job.isfile(rdf_solo_file)

@Project.label
def rdf_done(job):
    return job.isfile(rdf_file)

@Project.label
def conductivity_done(job):
    return job.isfile(conductivity_file)

@Project.label
def conductivity_solo_done(job):
    return job.isfile(conductivity_solo_file)

@Project.label
def cn_done(job):
    return job.isfile(cn_file)

@Project.label
def cn_solo_done(job):
    return job.isfile(cn_solo_file)

@Project.label
def velocity_done(job):
    return job.isfile(velocity_file)

@Project.label
def gk_done(job):
    return job.isfile(gk_file)


@Project.operation
@Project.post.isfile(init_file)
def initialize(job, scale = 1.24, for_lammps = False):

    import parmed as pmd
    print(job.id)
    print("Setting up packing ...")
    case = job.statepoint()["case"]  # mol/L , concIL is the concentration of either cation or anion
    print(case)

    # if case == 'neat_emimtfsi':
    #     volume_scale =1.109772667013316
    # elif case == 'acn_emimtfsi':
    #     volume_scale = 1.2374722028442868
    # elif case == 'acn_litfsi':
    #     volume_scale = 1.7
    # elif case == 'wat_litfsi':
    #     volume_scale = 1.4805186139912172


    print('case is {}, volume_scale is {}'.format(case, scale))
    MW = {
            "acn": 41.05,
            "li_tfsi": 287.09,
            "emim_tfsi": 391.31,
            "bmim_tfsi": 419.36,
            "wat": 18.01528
        }  # g/mol
    ###
    density = {
        "acn": 0.786,  #* density_scale, # real is 0.786
        "li_tfsi": 1.33,  # real is 1.33,
        "emim_tfsi": 1.5235,
        "bmim_tfsi": 1.4431,
        "wat": 0.997,
    }  # g/cm^(3)

    box_lengths = [3.5, 6.86, 3.5]
    ff_file_path = './cdc_xml_trj/cdc.xml'
    trj_file_path = './cdc_xml_trj/generate.xtc'
    n_atoms = 4000

    general_builder = model_builder.builder()
    pore_length = box_lengths[2]
    pore_depth = box_lengths[0]

    cdc = general_builder.cdc(box_lengths, ff_file_path, trj_file_path, n_atoms)
    graphene = general_builder.make_graphene(ff_file_path, pore_length, pore_depth, 1)
    general_builder.rotate(axis=0, target='graphene')
    limit = 9
    general_builder.remove_partial_cdc(axis = 1, limit = limit)
    move_distance = -limit + graphene.box[1]/3
    general_builder.move_structure(1, move_distance, 'cdc')
    combined_cdc = general_builder.combine()
    for res in combined_cdc.residues:
        res.name = 'cdc'
    print("combine_cdc has been finished")
    
    # combined_cdc.save('structure.gro', overwrite = True)
    combined_cdc_length = cdc.box[1]/10 + move_distance/10## unit is nm
    shorten_length = 0
    packing_box = mb.Box([cdc.box[0]/10, combined_cdc_length * 2 - shorten_length, cdc.box[2]/10])
    ### volume is the volume for the solution, nm^3; I times 1.5 in order to put 1.5 times molecule number
    volume = packing_box.Lx * packing_box.Ly * packing_box.Lz
    anion = GetIL("tfsi")
    anion.name = "tfsi"
    if case == 'neat_emimtfsi':
        target = ['emim_tfsi']
        cation = GetIL("emim")
        cation.name = "emim"
        sol_ratio = 0
        # guess_ratio = 2.0892162213214713
        n_list = general_builder.calculate_n(target, MW, density, volume)
    if case == 'acn_emimtfsi':
        target = ['acn', 'emim_tfsi']
        cation = GetIL("emim")
        cation.name = "emim"
        sol = GetIL("acn")
        sol.name = "acn"
        sol_ratio = 3
        # guess_ratio = 1.2211215862421407
        n_list = general_builder.calculate_n(target, MW, density, volume, sol_ratio)
    if case == 'acn_litfsi':
        target = ['acn', 'li_tfsi']
        cation = GetIL("li")
        cation.name = "li"
        sol = GetIL("acn")
        sol.name = "acn"
        sol_ratio = 3
        # guess_ratio = 1.6598848726544895
        n_list = general_builder.calculate_n(target, MW, density, volume, sol_ratio)
    if case == 'wat_litfsi':
        target = ['wat', 'li_tfsi']
        cation = GetIL("li")
        cation.name = "li"
        sol = GetIL("water")
        sol.name = "wat"
        sol_ratio = 3
        # guess_ratio = 2.602653204330833
        n_list = general_builder.calculate_n(target, MW, density, volume, sol_ratio)
    
    # if case == 'neat_emimtfsi':
    #     scale =  1.52
    # if case == 'acn_litfsi':
    #     scale = 2.3
    # scale  = 1.73
    print('scale is {}'.format(scale))
    # n_list = np.array([int(guess_ratio * volume * sol_ratio), int(guess_ratio * volume), int(guess_ratio * volume)])
    n_list = np.array(n_list)
    n_list = n_list * scale 
    n_list = n_list.astype(int)
    if for_lammps:
        solvent_n, cation_n, anion_n, other_position = desired_res_n_lmp(job)
        n_list = [solvent_n, cation_n, anion_n]
        if case == 'neat_emimtfsi':
            n_list = n_list[-2:]
        print("scale is not usefull, the case is {}, n_list is {}".format(case,n_list))
    if case != 'neat_emimtfsi':
        print("start to build solution box system and n_list is {}".format(n_list))
        print('current case is {} and packing box is {}'.format(case, packing_box))
        box_system = mb.fill_box(
                                    compound = [sol, cation, anion],
                                    n_compounds = [
                                                int(n_list[0]),
                                                int(n_list[1]),
                                                int(n_list[1])
                                                ],
                                    box = packing_box,
                                    seed = int(job.statepoint()["seed"] * 3000),
                                    edge=0.05
                                    )
        cation_cmp = mb.Compound()
        anion_cmp = mb.Compound()
        sol_cmp = mb.Compound()

        for child in box_system.children:
            if child.name == 'acn' or child.name == 'wat':
                sol_cmp.add(mb.clone(child))
            elif child.name == 'li' or child.name == 'emim':
                cation_cmp.add(mb.clone(child))
            elif child.name == 'tfsi':
                anion_cmp.add(mb.clone(child))


        clp = ilff.load_LOPES()
        anionPM =clp.apply(anion_cmp, residues = 'tfsi')
        if cation.name == 'li':
            opls_li = Get_ff_path('opls_ions')
            opls_li = Forcefield(opls_li)
            cationPM = opls_li.apply(cation_cmp, residues = 'li')
        if cation.name == 'emim':
            cationPM = clp.apply(cation_cmp, residues = "emim")
        if sol.name == 'acn':
            opls = Forcefield(name="oplsaa")
            solPM = opls.apply(sol_cmp, residues = 'acn')
        if sol.name == 'wat':
            spce = Get_ff_path('spce')
            spce = Forcefield(spce)
            solPM = spce.apply(sol_cmp, residues = 'wat')

        sol_structure = solPM + cationPM + anionPM
    if case == 'neat_emimtfsi':
        print("start to build solution box system and n_list is {}".format(n_list))
        print('current case is {} and packing box is {}'.format(case, packing_box))
        box_system = mb.fill_box(
                                    compound = [cation, anion],
                                    n_compounds = [
                                                int(n_list[0]),
                                                int(n_list[0])
                                                ],
                                    box = packing_box,
                                    seed = int(job.statepoint()["seed"] * 3000),
                                    edge=0.05
                                    )
        cation_cmp = mb.Compound()
        anion_cmp = mb.Compound()

        for child in box_system.children:
            if child.name == 'emim':
                cation_cmp.add(mb.clone(child))
            elif child.name == 'tfsi':
                anion_cmp.add(mb.clone(child))

        clp = ilff.load_LOPES()
        anionPM = clp.apply(anion_cmp, residues = 'tfsi')
        cationPM = clp.apply(cation_cmp, residues = "emim")

        sol_structure =  cationPM + anionPM

    ### move sol_structure
    for atom in sol_structure.atoms:
        atom.xy += combined_cdc_length *10

    ### match up the other atom positions with sample last frame positions
    if for_lammps:
        solvent_n, cation_n, anion_n, other_position = desired_res_n_lmp(job)
        i = 0 
        for atom in sol_structure.atoms:
            atom.xx = other_position[i][0]
            atom.xy = other_position[i][1]
            atom.xz = other_position[i][2]
            i += 1

    print('start to build symmetric cdc system')
    combined_cdc_copy = combined_cdc.__copy__()
    middle_line = combined_cdc_length * 10 * 2 - shorten_length/2 *10
    for atom in combined_cdc.atoms:
        distance_to_middle_line = middle_line  - atom.xy
        atom.xy += 2*distance_to_middle_line
    print("start to combine structure")
    structure = combined_cdc_copy + combined_cdc + sol_structure

    for atom in structure.atoms:
        if atom.residue.name in ['li', 'tfsi', 'fsi', 'emim']:
            atom.charge *= 0.8
            
    structure.box[0] = cdc.box[0]
    situation = 'main'
    if situation == 'main':
        structure.box[1] = middle_line * 2 * 2
    if situation == 'test':
        structure.box[1] = middle_line * 2 + 1
    structure.box[2] = cdc.box[2]
    structure.combining_rule = 'geometric'
    
    if for_lammps:
        print("saving gro and top file for lammps")
        structure.save(os.path.join(job.workspace(), "system_lmp.gro"), combine = 'all', overwrite=True)
        structure.save(os.path.join(job.workspace(), "system_lmp.top"), overwrite=True)
        print('start to rotate and save to lammps format')
        structure.box[1] = middle_line * 2 + 1
        ## rotate for latter conp slab correction
        structure = general_builder.general_rotate(0, structure)
        lammpsdata.write_lammpsdata(structure, os.path.join(job.workspace(), "sample.data"))
        # lammpsdata.write_lammpsdata(structure, os.path.join(job.workspace(), "sample.data"), mins = [0, 0, 0],
        #                     maxs = [structure.box[0]/10, combined_cdc_length * 10 * 4 /10, structure.box[2]/10])
    else:
        print("saving gro and top file")
        structure.save(os.path.join(job.workspace(), "system.gro"), combine = 'all', overwrite=True)
        structure.save(os.path.join(job.workspace(), "system.top"), overwrite=True)

def desired_res_n_lmp(job):
    trj_file = os.path.join(job.workspace(), 'sample.xtc')
    tpr_file = os.path.join(job.workspace(), 'system.gro')
    universe = mda.Universe(tpr_file, trj_file)
    cdc_atoms = universe.select_atoms('resname cdc')
    solvent_atoms = universe.select_atoms('resname {}'.format(job.statepoint()['case'][:3]))

    all_atoms = universe.select_atoms('all')
    other_atoms = universe.select_atoms('not resname cdc')
    ### get the positions for atoms except cdc in the last frame; note you need : after -1
    other_position = [other_atoms.positions for ts in universe.trajectory[-1:]][0]
    cation_n = int((other_atoms.n_residues - solvent_atoms.n_residues)/2)
    anion_n = cation_n
    solvent_n = solvent_atoms.n_residues
    return solvent_n, cation_n, anion_n, other_position

@Project.operation
# @Project.pre.isfile(init_file)
@Project.post.isfile(auto_calibrate_file)
def auto_calibrate(job):
    ### remove old calibrate files
    import subprocess
    remove = 'rm *calibrate* && rm \#*'
    subprocess.run(workspace_command_auto(
        remove, job
    ),
    shell=True, stdout=subprocess.PIPE, universal_newlines=True
    )
    ## variable to change temperature
    Temp = 400
    ## auto calibrate the bulk density besides the porous structure
    case = job.statepoint()["case"] 
    process_file_path = './txt_files/process'
    process_file = open('{}/auto_calibrate_{}.txt'.format(process_file_path, case), 'w')
    print("hello, autocalibration starts... \n", file = process_file)
    process_file.close()
    
    path_bulk = '/raid6/homes/linx6/project/self_project/cdc_bulk_clp_1fs_lopesbondfix/txt_files/density'
    bulk_data = np.loadtxt('{}/bulk_density_{}_{}K.txt'.format(path_bulk, case, Temp))
    bulk_density = np.mean(bulk_data[:,1])

    #### first calibrate to get cdc density data
    scale = 1.24
    
    if case == "acn_emimtfsi":
        # scale = 1.49
        scale = 1.446
    if case == "acn_litfsi":
        # scale = 1.89
        scale = 1.781
    if case == "wat_litfsi":
        scale = 1.9549155439561288
    if case == "neat_emimtfsi":
        scale = 1.287
    i = 0
    process_file = open('{}/auto_calibrate_{}.txt'.format(process_file_path, case), 'a', buffering=1)
    process_file.write('initializing the system {}, using the scale {}, this is the time {} \n'.format(case, scale, i))
    process_file.close()
    initialize(job, scale)
    process_file = open('{}/auto_calibrate_{}.txt'.format(process_file_path, case), 'a', buffering=1)
    process_file.write('starting to calibrate the system, long time warning! \n')
    process_file.close()
    calibrate_step_new("em", "nvt1_calibrate", "nvt2_calibrate", 'sample', Temp, job)
    process_file = open('{}/auto_calibrate_{}.txt'.format(process_file_path, case), 'a', buffering=1)
    process_file.write('starting to calculate density \n')
    process_file.close()
    density_gmx(job)

    trj_file = os.path.join(job.workspace(), 'sample.xtc')
    tpr_file = os.path.join(job.workspace(), 'sample.tpr')
    universe = mda.Universe(tpr_file, trj_file)
    cdc = universe.select_atoms('resname cdc')
    ### figure out the y value for the center between two cdc systems
    y_center = (np.max(cdc.positions[:,1]) + np.min(cdc.positions[:,1]))/2

    ### path to the cdc density files
    path = './txt_files/density'
    data = np.loadtxt('{}/cdc_density_{}.txt'.format(path, case))
    ### select index and calculate middle density for middle region
    half_range = 3.5
    select_index = np.where((data[:,0] < y_center/10 + half_range) &(data[:,0] > y_center/10 - half_range))
    middle_density_list = data[select_index[0], 1]
    avg_middle_density = np.mean(middle_density_list)

    process_file = open('{}/auto_calibrate_{}.txt'.format(process_file_path, case), 'a', buffering=1)
    process_file.write('The current avg_middle_density and bulk density are {} and {} \n \n \n'.format(avg_middle_density, bulk_density))
    process_file.close()

    
    
    # if case == 'acn_emimtfsi':
    #     upper_scale = 1.23
    #     lower_scale = 1.19
    # elif case == 'acn_litfsi':
    #     upper_scale = 1.23
    #     lower_scale = 1.17
    # else:
    upper_scale = 1.12
    lower_scale = 1.090
        
    while (avg_middle_density > bulk_density * upper_scale) or (avg_middle_density < bulk_density * lower_scale):
        i = i + 1
        middle_scale = (upper_scale + lower_scale)/2
        new_scale = bulk_density * middle_scale / avg_middle_density
        scale = scale * new_scale
        process_file = open('{}/auto_calibrate_{}.txt'.format(process_file_path, case), 'a', buffering=1)
        process_file.write('initializing the system {}, using the scale {}, this is the time {} \n'.format(case, scale, i))
        process_file.close()
        initialize(job, scale)
        process_file = open('{}/auto_calibrate_{}.txt'.format(process_file_path, case), 'a', buffering=1)
        process_file.write('starting to calibrate the system, long time warning! \n')
        process_file.close()
        calibrate_step_new("em", "nvt1_calibrate", "nvt2_calibrate", 'sample', Temp, job)
        process_file = open('{}/auto_calibrate_{}.txt'.format(process_file_path, case), 'a', buffering=1)
        process_file.write('starting to calculate density \n')
        process_file.close()
        density_gmx(job)
        data = np.loadtxt('{}/cdc_density_{}.txt'.format(path, case))
        ### select index and calculate middle density for middle region
        select_index = np.where((data[:,0] < y_center/10 + half_range) &(data[:,0] > y_center/10 - half_range))
        middle_density_list = data[select_index[0], 1]
        avg_middle_density = np.mean(middle_density_list)
        process_file = open('{}/auto_calibrate_{}.txt'.format(process_file_path, case), 'a', buffering=1)
        process_file.write('The current avg_middle_density and bulk density are {} and {} \n \n \n'.format(avg_middle_density, bulk_density))
        process_file.close()
        
    process_file = open('{}/auto_calibrate_{}.txt'.format(process_file_path, case), 'a', buffering=1)
    process_file.write('yes, you are done. The current avg_middle_density and bulk density ' + 
             'are {} and {} and the scale is {} and repeating time (including 0) is {} \n'.format(avg_middle_density, bulk_density, scale, i+1))
    process_file.close()
    np.savetxt(os.path.join(job.workspace(), 'auto_calibration.txt'), [scale])

    ### remove files that start with # to reduce the size of workspace
    remove_file = os.path.join(job.workspace(), '\#*')
    os.system('rm {}'.format(remove_file))
    
    # np.savetxt(os.path.join(job.workspace(), 'calibrate_done.txt'), [1,1]) 
    # ### remove files that start with # to reduce the size of workspace
    # remove_file = os.path.join(job.workspace(), '\#*')
    # os.system('rm {}'.format(remove_file))


def calibrate_step_new(op_name_em, op_name_1, op_name_2, op_name_3, T, job):
    system_top_file = os.path.join(job.workspace(), 'system.top')
    system_gro_file = os.path.join(job.workspace(), 'system.gro')

    mdp_em = signac.get_project().fn('src/util/mdp_files/{}.mdp'.format(op_name_em))
    mdp1 = signac.get_project().fn('src/util/mdp_files/{}-{}.mdp'.format(op_name_1, T))
    mdp2 = signac.get_project().fn('src/util/mdp_files/{}-{}.mdp'.format(op_name_2, T))
    mdp3 = signac.get_project().fn('src/util/mdp_files/{}-{}.mdp'.format(op_name_3, T))
    import subprocess
    # subprocess.run('cd workspace/{}/'.format(job.id), shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
    em = 'gmx grompp -f {mdp_em} -c {system_gro} -p {system_top} -o {op_em}.tpr --maxwarn 1 && gmx mdrun -deffnm {op_em} -ntmpi 1'
    subprocess.run(workspace_command_auto(em.format(mdp_em=mdp_em, op_em = op_name_em, 
                                                   system_gro=system_gro_file, system_top = system_top_file),
                                                   job),
                                                   shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
    step1 = 'gmx grompp -f {mdp1} -c {gro1}.gro -p {system_top} -o {op1}.tpr --maxwarn 1 && gmx mdrun -deffnm {op1} -ntmpi 1'
    subprocess.run(workspace_command_auto(
                        step1.format(
                            mdp1=mdp1, op1 = op_name_1, gro1 = op_name_em,
                            system_gro=system_gro_file, system_top = system_top_file
                        ), 
                        job), 
                        shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True
                    )
    step2 = 'gmx grompp -f {mdp2} -c {gro2}.gro -p {system_top} -o {op2}.tpr --maxwarn 1 && gmx mdrun -deffnm {op2} -ntmpi 1'   
    subprocess.run(workspace_command_auto(
                        step2.format(
                            mdp2=mdp2, op2 = op_name_2, gro2 = op_name_1,
                            system_gro=system_gro_file, system_top = system_top_file
                        ), 
                        job), 
                        shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True
                    )
    step3 = 'gmx grompp -f {mdp3} -c {gro3}.gro -p {system_top} -o {op3}.tpr --maxwarn 1 && gmx mdrun -deffnm {op3} -ntmpi 1'
    subprocess.run(workspace_command_auto(
                        step3.format(
                            mdp3=mdp3, op3 = op_name_3, gro3 = op_name_2,
                            system_gro=system_gro_file, system_top = system_top_file
                        ), 
                        job), 
                        shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True
                    )
    


def calibrate_step(op_name_em, op_name_1, op_name_2, op_name_3, T, job):
    system_top_file = os.path.join(job.workspace(), 'system.top')
    system_gro_file = os.path.join(job.workspace(), 'system.gro')

    mdp_em = signac.get_project().fn('src/util/mdp_files/{}.mdp'.format(op_name_em))
    mdp1 = signac.get_project().fn('src/util/mdp_files/{}-{}.mdp'.format(op_name_1, "600"))
    mdp2 = signac.get_project().fn('src/util/mdp_files/{}-{}.mdp'.format(op_name_2, T))
    mdp3 = signac.get_project().fn('src/util/mdp_files/{}-{}.mdp'.format(op_name_3, T))

    cmd = (
            'gmx grompp -f {mdp_em} -c {system_gro} -p {system_top} -o {op_em}.tpr --maxwarn 1 '\
            '&& gmx mdrun -deffnm {op_em} -ntmpi 1 '\
            '&& gmx grompp -f {mdp1} -c {gro1}.gro -p {system_top}  -o {op1}.tpr --maxwarn 1 '\
            '&& gmx mdrun -deffnm {op1} -ntmpi 1 '\
            '&& gmx grompp -f {mdp2} -c {gro2}.gro -p {system_top}  -o {op2}.tpr --maxwarn 1 '\
            '&& gmx mdrun -deffnm {op2} -ntmpi 1 '\
            '&& gmx grompp -f {mdp3} -c {gro3}.gro -p {system_top}  -o {op3}.tpr --maxwarn 1 '\
            '&& gmx mdrun -deffnm {op3} -ntmpi 1'
            )

    os.system(workspace_command_auto(cmd.format(mdp_em=mdp_em, op_em = op_name_em, 
                        mdp1=mdp1, op1 = op_name_1, gro1 = op_name_em,
                        mdp2=mdp2, op2 = op_name_2, gro2 = op_name_1,
                        mdp3=mdp3, op3 = op_name_3, gro3 = op_name_2,
                        system_gro=system_gro_file, system_top = system_top_file
                        ), job))

def workspace_command_auto(cmd, job):
    """Simple command to always go to the workspace directory"""
    return " && ".join(
        [
            "cd workspace/{}".format(job.id),
            cmd if not isinstance(cmd, list) else " && ".join(cmd),
            "cd ../..",
        ]
    )


# os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -pbc nojump'.format(xtc_file, unwrapped_trj, gro_file))



@Project.operation
@Project.pre.isfile(init_file)
@Project.post.isfile(em_file)
@flow.cmd
def em(job):
    return _gromacs_str("em", "init", "init", job)


@Project.operation
@Project.pre.isfile(em_file)
@Project.post.isfile(nvt1_file)
@flow.cmd
def nvt1(job):
    return _gromacs_str("nvt1_calibrate", "em", "init", job)


@Project.operation
@Project.pre.isfile(nvt1_file)
@Project.post.isfile(nvt2_file)
@flow.cmd
def nvt2(job):
    return _gromacs_str("nvt2", "nvt1", "init", job)



@Project.operation
@Project.pre.isfile(nvt2_file)
@Project.post.isfile(sample_file)
@flow.cmd
def sample(job):
    return _gromacs_str("sample", "nvt2", "init", job)


@Project.operation
@Project.pre.isfile(nvt2_file)
@Project.post.isfile(sample1_file)
@flow.cmd
def sample1(job):
    return _gromacs_str("sample1", "nvt2", "init", job)

@Project.operation
@Project.pre.isfile(init_file)
@Project.post.isfile(sample_file)
@flow.cmd
def sample_total(job):
    return _gromacs_str_total("em", "nvt1", "nvt2", 'sample', '298', job)

@Project.label
def for_lammpsed(job):
    return job.isfile(for_lammps_data)
for_lammps_data = 'sample.data'
@Project.operation
@Project.pre.isfile(sample_file)
@Project.post.isfile(for_lammps_data)
def for_lammps(job):
    case = job.statepoint()["case"] 
    print(case)
    ### scale is not usefull, when use for_lammps = True
    initialize(job, for_lammps = True)


@Project.operation
@Project.post.isfile(lammps_init_file)
@flow.cmd
def run_copy_lmpinit(job):
    return copy_file(job)

def copy_file(job):
    """Helper function, returns lammps command string for operation """
    case = job.statepoint()["case"] 
    print(case)
    print(job.id)
    if case == 'neat_emimtfsi':
        data_file = '../../workspace/4ac1bd4c7366a76190efd348d33de45c/sample.data'
    if case == 'acn_emimtfsi':
        data_file = '../../workspace/6159b48c31cc80881b047d79feb31bb6/sample.data'
    if case == 'acn_litfsi':
        data_file = '../../workspace/571107cd66f5832bdaa64934a9e63c3c/sample.data'
    if case == 'wat_litfsi':
        data_file = '../../workspace/b5e865c4bcc06826d64c25344ca8d54a/sample.data'
    # regenerate_input_file_cmd = 'cd ../../lammps_input/ && python get_voltage.py && cd ../workspace/{}'.format(job.id)
    cmd = 'cp {data_file} .'
    return workspace_command(cmd.format(data_file=data_file))

@Project.operation
@Project.pre.isfile(lammps_init_file)
@Project.post.isfile(conp_file)
@flow.cmd
def run_cpm(job):
    return _lammps_str(job)

def _lammps_str(job, r_value=0, w_value =0, in_path = 'lammps_input/in.data', reset = 0):
    """Helper function, returns lammps command string for operation 
        Note: need to use cori_start.sh or cori_repeat.sh according to demand
    """
    case = job.statepoint()["case"] 
    voltage = job.statepoint()["voltage"] 
    lammps_input = signac.get_project().fn(in_path)
    Temp = 400
    print(case)
    print(job.id)
    if case == 'wat_litfsi':
        w_value = 1
    
    if r_value == 0:
        cmd ='export KMP_BLOCKTIME=0\n'\
            'export case={case}\n'\
            'export N={voltage}\n'\
            'export R={r_value}\n'\
            'export W={w_value}\n'\
            'export T={Temp}\n'\
            'export S={reset}\n'\
            'mpirun -np 32 /global/project/projectdirs/m1046/Xiaobo/installed_software/lammps_stable_May27_CPM/build/lmp_mpi -in {input} -sf intel -pk intel 0 omp 2'
    else:
        cmd ='export KMP_BLOCKTIME=0\n'\
            'export case={case}\n'\
            'export N={voltage}\n'\
            'export R={r_value}\n'\
            'export W={w_value}\n'\
            'export T={Temp}\n'\
            'export S={reset}\n'\
            'mpirun -np 32 /global/project/projectdirs/m1046/Xiaobo/installed_software/lammps_stable_May27_CPM/build/lmp_mpi -in {input} -sf intel -pk intel 0 omp 2 &\n'\
            'wait'
    return workspace_command(cmd.format(case = case, voltage=voltage, r_value= r_value, w_value= w_value, Temp = Temp, reset = reset, input = lammps_input))


sample_file = 'sample.data'
restart2500_file = 'file.restart.2500'
@Project.label
def rerun_cpm_reseted(job):
    return job.isfile(restart2500_file)

@Project.operation
@Project.pre.isfile(sample_file)
@Project.post.isfile(restart2500_file)
@flow.cmd
def rerun_cpm_reset(job):
    return _lammps_str(job, reset=1)


@Project.label
def rerun_cpmed(job):
    return job.isfile(conp_file)

@Project.operation
@Project.pre.isfile(restart2500_file)
@Project.post.isfile(conp_file)
@flow.cmd
def rerun_cpm(job):
    return _lammps_str(job, r_value = 1)

@Project.label
def com_bined(job):
    return job.isfile(com_file)

restart_file = 'file.restart.5000000'
com_file = "com_distribution.lammpstrj"
@Project.operation
@Project.pre.isfile(restart_file)
@Project.post.isfile(com_file)
@flow.cmd
def run_com_bin(job):
    return _lammps_str(job, in_path ='lammps_input/in.com_distribution')

# def _lammps_str_com_bin(job):
#     """Helper function, returns lammps command string for operation """
#     case = job.statepoint()["case"] 
#     voltage = job.statepoint()["voltage"] 
#     print(case)
#     print(job.id)
#     lammps_input = signac.get_project().fn('lammps_input/in.com_distribution')
#     cmd = 'source /raid6/homes/linx6/intel/oneapi/setvars.sh && '\
#         'export KMP_BLOCKTIME=0 && '\
#         'export case={case} && '\
#         'export V={voltage} && '\
#         'mpirun -np 16 /raid6/homes/linx6/install_software/lammps_stable_May27_CPM/build/lmp_mpi -in {input} -sf intel -pk intel 0 omp 2'
#     return workspace_command(cmd.format(input = lammps_input, case = case, voltage= voltage))

Project.label
def potential_atc_bined(job):
    return job.isfile(potential_atc_file)
new_dump_file = 'new_conp.lammpstrj'
potential_atc_file = "potential_atc_distribution.lammpstrj"
@Project.operation
@Project.pre.isfile(new_dump_file)
@Project.post.isfile(potential_atc_file)
@flow.cmd
def run_potential_atc_bin(job):
    return _lammps_str(job, in_path = 'lammps_input/in.potential_atc')

@Project.operation
@Project.pre.isfile(conpxtc_file)
@Project.post.isfile(unwrapped_file)
def prepare(job):
    xtc_file = os.path.join(job.workspace(), 'conp.xtc')
    gro_file = os.path.join(job.workspace(), 'system_lmp.gro')
    # tpr_file = os.path.join(job.workspace(), 'sample.gro')
    if os.path.isfile(xtc_file) and os.path.isfile(gro_file):
        unwrapped_trj = os.path.join(job.workspace(),
        'sample_unwrapped.xtc')
        os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -pbc nojump'.format(xtc_file, unwrapped_trj, gro_file))
        res_trj = os.path.join(job.ws, 'sample_res.xtc')
        com_trj = os.path.join(job.ws, 'sample_com.xtc')
        unwrapped_com_trj = os.path.join(job.ws,'sample_com_unwrapped.xtc')
        os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -pbc res'.format(
                xtc_file, res_trj, gro_file))
        trj = md.load(res_trj, top=gro_file)
        comtrj = make_comtrj(trj)
        comtrj.save_xtc(com_trj)
        comtrj[-1].save_gro(os.path.join(job.workspace(),
             'com.gro'))
        print('made comtrj ...')
        os.system('gmx trjconv -f {0} -o {1} -pbc nojump'.format(
                com_trj, unwrapped_com_trj))

@Project.operation
@Project.pre.isfile(sample_file)
@Project.post.isfile(density_gmx_file)
def density_gmx(job):
    xtc_file = os.path.join(job.workspace(), 'sample.xtc')
    tpr_file = os.path.join(job.workspace(), 'sample.tpr')
    xvg_file = os.path.join(job.workspace(), 'sample_density.xvg')
    os.system('echo 0 | gmx density -f {0} -s {1} -o {2} -d Y'.format(xtc_file, tpr_file, xvg_file))
    # aux_file = os.path.join(job.workspace(), 'sample_density.xvg')
    aux = mda.auxiliary.auxreader(xvg_file)
    data = np.array([step.data for step in aux])
    np.savetxt('./txt_files/density/cdc_density_{}.txt'.format(job.statepoint()['case']), data, header='# coordinate (nm) vs average density (kg/m^3)')


Project.label
def generate_xtc(job):
    return job.isfile(charge_time_file)

result_file = 'conp.lammpstrj'
charge_time_file = "record_xtc.lammpstrj"
@Project.operation
@Project.pre.isfile(result_file)
@Project.post.isfile(charge_time_file)
@flow.cmd
def run_generate_xtc(job):
    return _lammps_str_xtc(job)

def _lammps_str_xtc(job):
    """Helper function, returns lammps command string for operation """
    case = job.statepoint()["case"] 
    voltage = job.statepoint()["voltage"] 
    seed = job.statepoint()["seed"] 
    print(case)
    print(job.id)
    if seed == 3:
        start = 2000000
        end = 12000000
    else:
        start = 50000
        end = 10050000
    lammps_input = signac.get_project().fn('lammps_input/in.generate_xtc')
    cmd = 'source /global/project/projectdirs/m1046/Xiaobo/installed_software/intel/oneapi/setvars.sh && '\
        'export KMP_BLOCKTIME=0 && '\
        'export start={start} && '\
        'export end={end} && '\
        'mpirun -np 32 /global/project/projectdirs/m1046/Xiaobo/installed_software/lammps_stable_May27_CPM/build/lmp_mpi -in {input} -sf intel -pk intel 0 omp 2'
    return workspace_command(cmd.format(input = lammps_input, start=start, end = end))


q_file = "q.txt"
Project.label
def save_q(job):
    return job.isfile(q_file)

result_file = 'new_conp.lammpstrj'

@Project.operation
@Project.pre.isfile(result_file)
@Project.post.isfile(q_file)
def run_save_q(job):
    ### clean original dump file
    import time
    from CPManalysis.clean_file import clean_dumpfile
    from CPManalysis.read_file import q_np
    tic = time.perf_counter()
    print(job.id,flush=True)
    print("start to clean and create new new_conp.lammpstrj, 2.5 min", flush=True)
    original_trj_file = os.path.join(job.ws, 'conp.lammpstrj')
    if job.statepoint()['seed'] != 3:
        new_content = clean_dumpfile(original_trj_file, stop_at = 12701000)
    else:
        new_content = clean_dumpfile(original_trj_file)
    trj_file = os.path.join(job.ws, 'new_conp.lammpstrj')
    text_file = open(trj_file, "w")
    n = text_file.write(new_content)
    text_file.close()
    toc = time.perf_counter()
    print(f"total time of clean conp.lammpstrj is {toc - tic:0.4f} seconds", flush=True)
    ### save q
    toc = time.perf_counter()
    print('start to save charge.npy file, 7 min', flush=True)
    
    top_file = os.path.join(job.ws, 'system_lmp.gro')
    u = mda.Universe(top_file)
    n_atom = u.atoms.n_atoms
    charge_2d = q_np(trj_file, n_atom)
    charge_file = os.path.join(job.ws, 'charge.npy')
    np.save(charge_file, charge_2d)
    np.savetxt(os.path.join(job.workspace(), 'q.txt'), [1,1])
    print('it is done', flush=True)
    toc = time.perf_counter()
    print(f"total time of saving q is {toc - tic:0.4f} seconds", flush=True)

    ### save xtc 
    # print('start to save conp.xtc file, 15 min', flush=True)
    # trj = md.load(trj_file, top=top_file)
    # save_file = os.path.join(job.ws, 'conp.xtc')
    # trj.save(save_file)
    # np.savetxt(os.path.join(job.workspace(), 'q_xtc.txt'), [1,1])
    

    
    

Project.label
def save_xtc(job):
    return job.isfile(conpxtc_file)

@Project.operation
@Project.pre.isfile(new_conp_file)
@Project.post.isfile(conpxtc_file)
def run_save_xtc(job):
    ### save xtc 
    
    print('start to save conp.xtc file, 25 min', flush=True)
    trj_file = os.path.join(job.ws, 'new_conp.lammpstrj')
    top_file = os.path.join(job.ws, 'system_lmp.gro')
    trj = md.load(trj_file, top=top_file)
    save_file = os.path.join(job.ws, 'conp.xtc')
    trj.save(save_file)
    np.savetxt(os.path.join(job.workspace(), 'xtc.txt'), [1,1])
    print('it is done', flush=True)

Project.label
def save_pele_q(job):
    return job.isfile(pele_q_file)

pele_q_file = "pele_q_file.txt"
@Project.operation
@Project.pre.isfile(q_file)
@Project.post.isfile(pele_q_file)
def run_save_pele_q(job):
    ### produce new charge.npy files with discarded frames and atom charge summation calculation for positive electrode
    print('start to save pele_q', flush=True)
    charge_file = os.path.join(job.workspace(), "charge.npy")
    charge = np.load(charge_file)
    if job.statepoint()['seed'] ==3:
        discard_frame = 2000
    else:
        discard_frame = 50
    charge = charge[discard_frame:]
    
    ### new_charge.npy is the atom charge after discarded frames
    new_charge_file = os.path.join(job.ws, 'new_charge.npy')
    np.save(new_charge_file, charge)
    gro_file = os.path.join(job.workspace(), "system_lmp.gro")
    gro_trj = md.load(gro_file)
    
    pos_ele = gro_trj.top.select('residue 1') ## IMportant: residue in mdtraj equal to resid in VMD and in gro file
    pos_trj = gro_trj.atom_slice(pos_ele)
    pos_charge = charge[:,pos_ele]
    sum_pos_q = np.sum(pos_charge, axis =1)
    xdata = np.arange(0, len(sum_pos_q),1)
    xdata = xdata * 0.002
    post_pele_charge = np.stack((xdata, sum_pos_q), axis = 1)
    
    ### pele_charge.npy is summed charge in positive electrode
    pele_charge_file = os.path.join(job.ws, 'pele_charge.npy') ### 
    np.save(pele_charge_file, post_pele_charge)
    np.savetxt(os.path.join(job.workspace(), 'pele_q_file.txt'), [1,1])
    print('it is done')

@Project.label
def ion_exchange(job):
    return job.isfile(ion_exchange_file)

ion_exchange_file = "ion_exchange.txt"
@Project.operation
@Project.pre.isfile(unwrapped_file)
@Project.post.isfile(ion_exchange_file)
def run_ion_exchange(job):
    ### only for 12 slices in one cdc electrode
    
    import CPManalysis.density as density
    print( job.statepoint()['case'])
    unwrapped_com_trj = os.path.join(job.ws,'sample_com.xtc')
    gro_com_file = os.path.join(job.workspace(), 'com.gro')
    trj_com = md.load(unwrapped_com_trj,stride=1, top=gro_com_file)
   
    gro_file = os.path.join(job.workspace(), 'system_lmp.gro')
    one_trj = md.load(gro_file)
    left_boundary = density.find_cdc_boundary(one_trj, 0)
    right_boundary = density.find_cdc_boundary(one_trj, 2)
    print(right_boundary)
    print(left_boundary)
    print(one_trj.unitcell_lengths)
    left = np.linspace(0, left_boundary,12)
    right = np.linspace(right_boundary,one_trj.unitcell_lengths[0][2],12)
    bins=  np.concatenate((left,right))
    # bins = [0, left_boundary-1, left_boundary, right_boundary, right_boundary+1, one_trj.unitcell_lengths[0][2]]
    for res_name in ['emim', 'tfsi', 'li','acn', 'wat']:
        try:
            print(res_name)
            trj_com_slice = trj_com.atom_slice(trj_com.top.select("resname {}".format(res_name)))
            print(res_name)
            new_t, mean_value = density.chunk_mean(trj_com_slice, bins=bins)
            new_data = np.hstack((new_t.reshape((-1,1)), mean_value.T))
            # print(new_data)
            data_file = os.path.join(job.ws, 'ion_exchange_{}.npy'.format(res_name))
            np.save(data_file, new_data)
        except:
            continue
        
    np.savetxt(os.path.join(job.workspace(), "ion_exchange.txt"), [1,1])


@Project.label
def ion_exchange_more(job):
    return job.isfile(ion_exchange_more_file)

ion_exchange_more_file = "ion_exchange_more.txt"
@Project.operation
@Project.pre.isfile(unwrapped_file)
@Project.post.isfile(ion_exchange_more_file)
def run_ion_exchange_more(job):
    ### only for many slices in one cdc electrode
    
    import CPManalysis.density as density
    print( job.statepoint()['case'])
    unwrapped_com_trj = os.path.join(job.ws,'sample_com.xtc')
    gro_com_file = os.path.join(job.workspace(), 'com.gro')
    trj_com = md.load(unwrapped_com_trj,stride=1, top=gro_com_file)
   
    gro_file = os.path.join(job.workspace(), 'system_lmp.gro')
    one_trj = md.load(gro_file)
    left_boundary = density.find_cdc_boundary(one_trj, 0)
    right_boundary = density.find_cdc_boundary(one_trj, 2)
    print(right_boundary)
    print(left_boundary)
    print(one_trj.unitcell_lengths)
    left = np.linspace(0, left_boundary,62)
    right = np.linspace(right_boundary,one_trj.unitcell_lengths[0][2],62)
    bins=  np.concatenate((left,right))
    # bins = [0, left_boundary-1, left_boundary, right_boundary, right_boundary+1, one_trj.unitcell_lengths[0][2]]
    for res_name in ['emim', 'tfsi', 'li','acn', 'wat']:
        try:
            print(res_name)
            trj_com_slice = trj_com.atom_slice(trj_com.top.select("resname {}".format(res_name)))
            print(res_name)
            new_t, mean_value = density.chunk_mean(trj_com_slice, bins=bins)
            new_data = np.hstack((new_t.reshape((-1,1)), mean_value.T))
            # print(new_data)
            data_file = os.path.join(job.ws, 'ion_exchange_{}_more.npy'.format(res_name))
            np.save(data_file, new_data)
        except:
            continue
        
    np.savetxt(os.path.join(job.workspace(), "ion_exchange_more.txt"), [1,1])
    
    
@Project.label
def z_dir_dist(job):
    return job.isfile(charge_com_file)

charge_com_file = "charge_com_dist.txt"
@Project.operation
@Project.pre.isfile(unwrapped_file)
@Project.post.isfile(charge_com_file)
def run_z_dir_dist(job):  
    ### charge and com density distribution in one direction
    
    
    ### charge distribution in cdc in one direction
    gro_file = os.path.join(job.workspace(), "system_lmp.gro")
    from CPManalysis.charge_dist import charge_dist
    case = job.statepoint()["case"]
    voltage = job.statepoint()["voltage"]
    seed = job.statepoint()["seed"]
    new_bins, avg_seed_charge_dist = charge_dist(case, voltage, gro_file, seeds = [seed], statistic='mean')
    new_data = np.hstack((new_bins.reshape((-1,1)), avg_seed_charge_dist.reshape((-1,1))))
    data_file = os.path.join(job.ws, 'charge_z_dist.npy')
    np.save(data_file, new_data)
    print('saved charge dist')
    ### COM density distribution in one direction
    from CPManalysis.density_dist import calc_density_distribution
    from CPManalysis.density import find_cdc_boundary
    structure = 'cdc'
    if structure == 'cdc':
        select_id = [0,2]
        prefix = ''
        gap = 0
    one_gro_file = os.path.join(job.workspace(), "{}system_lmp.gro".format(prefix))
    one_trj = md.load(one_gro_file)
    left_b = find_cdc_boundary(one_trj, select_id[0])
    right_b = find_cdc_boundary(one_trj, select_id[1])
    trj_file = os.path.join(job.workspace(), "{}sample_com_unwrapped.xtc".format(prefix))
    gro_file = os.path.join(job.workspace(), "{}com.gro".format(prefix))
    trj_total = md.load(trj_file,top = gro_file)
    if case == 'acn_litfsi' or case == 'wat_litfsi':
        res_name_list = ['li', 'tfsi']
    else:
        res_name_list = ['emim', 'tfsi']
        
    for res_name in res_name_list:
        new_bins, new_hist = calc_density_distribution(trj_total, last_n_frame=4000, res_name = res_name, binwidth = 0.1)
        x = new_bins+gap
        y = new_hist
        com_data = np.hstack((x.reshape((-1,1)), y.reshape((-1,1))))
        data_file = os.path.join(job.ws, 'com_z_dist_{}.npy'.format(res_name))
        np.save(data_file, com_data)
    print('saved com dist')
    
    np.savetxt(os.path.join(job.workspace(), "charge_com_dist.txt"), [1,1])  
    
@Project.label
def plane_charge_dist(job):
    return job.isfile(plane_charge_file)

plane_charge_file = "plane_charge_dist.txt"
@Project.operation
@Project.pre.isfile(unwrapped_file)
@Project.post.isfile(plane_charge_file)
def run_plane_charge_dist(job):  
    
    ### charge distribution in cdc in one direction
    gro_file = os.path.join(job.workspace(), "plane_system_lmp.gro")
    from CPManalysis.charge_dist import charge_dist
    case = job.statepoint()["case"]
    voltage = job.statepoint()["voltage"]
    seed = job.statepoint()["seed"]
    new_bins, avg_seed_charge_dist = charge_dist(case, voltage, gro_file, charge_file_name='plane_charge.npy', target_res = 'gra', seeds = [seed], statistic='mean')
    new_data = np.hstack((new_bins.reshape((-1,1)), avg_seed_charge_dist.reshape((-1,1))))
    data_file = os.path.join(job.ws, 'plane_charge_z_dist.npy')
    np.save(data_file, new_data)
    print('saved charge dist')
    np.savetxt(os.path.join(job.workspace(), "plane_charge_dist.txt"), [1,1])  
    
Project.label
def save_np_potential(job):
    return job.isfile(save_np_potential_file)

save_np_potential_file = "save_np_potential.txt"
@Project.operation
@Project.pre.isfile(potential_atc_file)
@Project.post.isfile(save_np_potential_file)
def run_save_np_potential(job):
    from CPManalysis.read_file import profile_reader
    potential_file = os.path.join(job.workspace(), "out.atc_potential3D.DATA")
    print(job.id, " start to read atc profile")
    data = profile_reader(potential_file, n_bins = 241)
    x = data[:,0]
    y = data[:,3]
    print('start to save npy')
    potential = np.stack((x, y), axis = 1)
    potential_file = os.path.join(job.ws, 'potential.npy') ### 
    np.save(potential_file, potential)
    
def _gromacs_str(op_name, gro_name, sys_name, job):
    """Helper function, returns grompp command string for operation """
    if op_name == 'em':
        mdp = signac.get_project().fn('src/util/mdp_files/{}.mdp'.format(op_name))
        cmd = ('gmx grompp -f {mdp} -c system.gro -p system.top -o {op}.tpr --maxwarn 1 && gmx mdrun -deffnm {op} -ntmpi 1')
    else:
        mdp = signac.get_project().fn('src/util/mdp_files/{}.mdp'.format(op_name))
        cmd = ('gmx grompp -f {mdp} -c {gro}.gro -p system.top -o {op}.tpr --maxwarn 1 && gmx mdrun -deffnm {op} -ntmpi 1')
    return workspace_command(cmd.format(mdp=mdp,op=op_name, gro=gro_name, sys=sys_name))


def _gromacs_str_total(op_name_em, op_name_1, op_name_2, op_name_3, T, job):
    """Helper function, returns grompp command string for operation """
    mdp_em = signac.get_project().fn('src/util/mdp_files/{}.mdp'.format(op_name_em))
    mdp1 = signac.get_project().fn('src/util/mdp_files/{}-{}.mdp'.format(op_name_1, T))
    mdp2 = signac.get_project().fn('src/util/mdp_files/{}-{}.mdp'.format(op_name_2, T))
    mdp3 = signac.get_project().fn('src/util/mdp_files/{}-{}.mdp'.format(op_name_3, T))
    cmd = (
            'gmx grompp -f {mdp_em} -c system.gro -p system.top -o {op_em}.tpr --maxwarn 1 '\
            '&& gmx mdrun -deffnm {op_em} -ntmpi 1 '\
            '&& gmx grompp -f {mdp1} -c {gro1}.gro -p system.top -o {op1}.tpr --maxwarn 1 '\
            '&& gmx mdrun -deffnm {op1} -ntmpi 1 '\
            '&& gmx grompp -f {mdp2} -c {gro2}.gro -p system.top -o {op2}.tpr --maxwarn 1 '\
            '&& gmx mdrun -deffnm {op2} -ntmpi 1 '\
            '&& gmx grompp -f {mdp3} -c {gro3}.gro -p system.top -o {op3}.tpr --maxwarn 1 '\
            '&& gmx mdrun -deffnm {op3} -ntmpi 1'
            )
    return workspace_command(cmd.format(mdp_em=mdp_em, op_em = op_name_em, 
                                        mdp1=mdp1, op1 = op_name_1, gro1 = op_name_em,
                                        mdp2=mdp2, op2 = op_name_2, gro2 = op_name_1,
                                        mdp3=mdp3, op3 = op_name_3, gro3 = op_name_2
                                        ))

if __name__ == "__main__":
    Project().main()
