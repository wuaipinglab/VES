import pandas as pd
import numpy as np
import scipy.stats
from sklearn.preprocessing import StandardScaler
from Bio.PDB import PDBParser
from weighted_contact_number import *
from seq_utils import *

####################################################################
# you may change to the new environment with lower biopython version 
####################################################################

##############################################
# General Paths
##############################################

# AA properties ()
aa_charge_hydro = './data/aa_properties/dissimilarity_metrics.csv' 

##############################################
# Flu Paths
##############################################

# Experimental data
h1_replication = './data/experiments/doud2016/Doud2016_h1_replication.csv'
h1_escape = './data/experiments/doud2018/DMS_Doud2018_H1-WSN33_antibodies.csv'
h1_experiment_range = (1, 565)


# Models
h1_eve = './results/mutation_fitness_prediction/aligned_single_processed_HA_60000_samples_FLU_DM_Unet_Attention_128_ch.csv'

# Structure data

h1_pdb_id = '1RVX'
h1_pdb_path = './data/structures/1rvx_no_HETATM.pdb'
h1_chains = ['A', 'B']
h1_trimer_chains = ['A', 'B', 'C', 'D', 'E', 'F']

h1_target_seq_path = './data/sequences/A0A2Z5U3Z0_9INFA.fasta'


##############################################
# Data Processing Functions
##############################################


def process_eve_smm(eve_path):
    '''
    Processes EVE single mutation matrix table
    '''
    eve = pd.read_csv(eve_path)
    eve = eve[1:]
    eve.columns = eve.columns.str.replace("_ensemble", "")
    eve['wt'] = eve.mutations.str[0]
    eve['mut'] = eve.mutations.str[-1]
    eve['i'] = eve.mutations.str[1:-1].astype(int)
    eve['model_prediction'] = -eve.model_prediction
    to_drop = ['protein_name', 'mutations']
    to_drop.extend([col for col in eve.columns if "semantic_change" in col])
    eve = eve.drop(columns=to_drop)
    return eve


def add_model_outputs(exps, eve_path):
    '''
    Merges EVE predictions on to experimental data table
    '''
    exps = exps.merge(process_eve_smm(eve_path),
                      on=['wt', 'mut', 'i'],
                      how='outer')
    return exps


def get_wcn(exps, pdb_path, trimer_chains, target_chains, map_table):
    '''
    Computes weighted contact number by alpha-carbon and sidechain 
    center of mass and merges on to experimental data table
    '''

    wcn = add_wcn_to_site_annotations(pdb_path, ''.join(trimer_chains))
    wcn = wcn.rename(columns={'pdb_position': 'i', 'pdb_aa': 'wt'})
    wcn['i'] = wcn.i.apply(lambda x: alphanumeric_index_to_numeric_index(x)
                           if (x != '') else x)
    wcn['i'] = wcn.i.replace('', np.nan)
    wcn = remap_struct_df_to_target_seq(wcn, target_chains, map_table)

    exps = exps.merge(wcn[['i', 'wcn_sc']], how='left', on='i')
    exps = exps.sort_values('i')
    exps['wcn_bfil'] = exps.wcn_sc.fillna(method='bfill')
    exps['wcn_ffil'] = exps.wcn_sc.fillna(method='ffill')
    exps['wcn_fill'] = (
        exps[['wcn_ffil', 'wcn_bfil']].sum(axis=1, min_count=2) / 2)
    exps = exps.drop(columns=['wcn_bfil', 'wcn_ffil'])
    return exps


def hydrophobicity_charge(exps, table):

    props = pd.read_csv(table, index_col=0)

    scale = StandardScaler()
    props['eisenberg_weiss_diff_std'] = scale.fit_transform(
        props['eisenberg_weiss_diff'].abs().values.reshape(-1, 1))
    props['charge_diff_std'] = scale.fit_transform(
        props['charge_diff'].abs().values.reshape(-1, 1))
    exps = exps.merge(props, how='left', on=['wt', 'mut'])

    exps['charge_ew-hydro'] = exps[[
        'eisenberg_weiss_diff_std', 'charge_diff_std'
    ]].sum(axis=1)
    exps = exps.drop(columns=['eisenberg_weiss_diff_std', 'charge_diff_std'])
    return exps


def norm_to_wt(df, prefvar):
    '''
    Normalize experimental variables to wildtype (for "prefs" style data) 
    '''
    newvar = 'norm_' + prefvar

    def grp_func(grp):

        ref = grp[grp['wt'] == grp['mut']][prefvar].mean()
        grp[newvar] = grp[prefvar] / ref
        return grp

    df[newvar] = df[prefvar]
    df = df.groupby(['i', 'wt']).apply(grp_func)
    return df

def rbd_metadata(escape_df, bloom_path, xie_path, metadata_path):
    escape = escape_df[['condition','condition_type',
                        'condition_subtype','condition_year',
                        'eliciting_virus','study',
                        'lab']].drop_duplicates()
    with open(xie_path, "w") as textfile:
        for element in escape[
                          (escape.lab=='Xie_XS')].condition.tolist():
            textfile.write(element + "\n")
    with open(bloom_path, "w") as textfile:
        for element in escape[
                          (escape.lab=='Bloom_JD')].condition.tolist():
            textfile.write(element + "\n")
    escape.to_csv(metadata_path)
    return(escape)


##############################################
# Summary workbook functions
##############################################


def load_flu_HA():

    # Read in and combine experimental data
    escape = pd.read_csv(h1_escape).drop(columns=['resi'])
    cols = ['wt', 'mut', 'i'] + [
        col for col in escape.columns if 'median_mutfracsurvive' in col
    ]
    escape = escape[cols]
    rep = pd.read_csv(h1_replication)
    rep = rep.rename(columns={'norm_tf_prefs': 'flu_h1_replication'})
    data = escape.merge(rep[['wt', 'mut', 'i', 'flu_h1_replication']],
                        on=['wt', 'mut', 'i'],
                        how='outer')

    # Read in and combine model data
    data = add_model_outputs(data, h1_eve)

    # Get rid of wt data
    data = data[data.wt != data.mut]

    # Get mapping to PDB
    map_table = remap_pdb_seq_to_target_seq(h1_pdb_path, h1_chains,
                                            h1_target_seq_path)

    # Calculated weighted contact counts
    data = get_wcn(data, h1_pdb_path, h1_trimer_chains, h1_chains, map_table)

    # Add aa properties to data
    data = hydrophobicity_charge(data, aa_charge_hydro)
    data = data.sort_values(['i', 'mut'])

    # Drop any rows not in experiment
    data = data[(data.i >= h1_experiment_range[0])
                & (data.i <= h1_experiment_range[1])]

    return data, map_table


ha, _ = load_flu_HA()
ha.to_csv('./results/prediction_w_exp/ves.csv', index=False)
