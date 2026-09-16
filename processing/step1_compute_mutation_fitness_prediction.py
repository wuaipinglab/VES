import os,sys
import json
import argparse
import pandas as pd
import torch
import numpy as np
import time
import tqdm
import torch.nn.functional as F

from training.Diffusion_Model_Unet_FLU_128_ch import UNetScoreBasedDiffusionModel
from utils import data_utils_FLU

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Mutation Fitness Prediction')
    parser.add_argument('--MSA_data_folder', default='./data/MSA', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--MSA_list', default='./data/mappings/example_mapping.csv', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', default=0, type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', default='./data/weights', type=str, help='Location where weights for each sequence in the MSA will be stored')
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--computation_mode', default='all_singles', type=str, help='Computes evol indices for all single AA mutations or for a passed in list of mutations (singles or multiples) [all_singles,input_mutations_list]')
    parser.add_argument('--all_singles_mutations_folder', default='./data/mutations', type=str, help='Location for the list of generated single AA mutations')
    parser.add_argument('--mutations_location', type=str, help='Location of all mutations to compute the evol indices for')
    parser.add_argument('--output_prediction_location', default='./results/mutation_fitness_prediction', type=str, help='Output location of computed evol indices')
    parser.add_argument('--output_prediction_filename_suffix', default='_FLU_DM_Unet_Attention_128_ch', type=str, help='(Optional) Suffix to be added to output filename')
    parser.add_argument('--num_samples_compute_predictions', default=60000, type=int, help='Num of samples to approximate delta elbo when computing evol indices')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size when computing evol indices')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mapping_file = pd.read_csv(args.MSA_list)
    protein_name = mapping_file['protein_name'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['msa_location'][args.protein_index]
    print("Protein name: "+str(protein_name))
    print("MSA file: "+str(msa_location))

    if args.theta_reweighting is not None:
        theta = args.theta_reweighting
    else:
        try:
            theta = float(mapping_file['theta'][args.protein_index])
        except:
            theta = 0.2
    print("Theta MSA re-weighting: "+str(theta))

    data = data_utils_FLU.MSA_processing(
            MSA_location=msa_location,
            theta=theta,
            use_weights=True,
            weights_location=args.MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'
    )
    
    if args.computation_mode=="all_singles":
        data.save_all_singles(output_filename=args.all_singles_mutations_folder + os.sep + protein_name + "_all_singles.csv")
        args.mutations_location = args.all_singles_mutations_folder + os.sep + protein_name + "_all_singles.csv"
    else:
        args.mutations_location = args.mutations_location + os.sep + protein_name + ".csv"
        

    model = UNetScoreBasedDiffusionModel(input_size=20,hidden_size=64,timesteps=1000,cond_dim=None).to(device)

    try:
        model.load_state_dict(torch.load('./results/Diffusion_model/FLU_DM_UNet_Attention_128_ch.pth'))
        print("Initialized model with checkpoint 'UNet_Attention_128_ch' ")
    except:
        print("Unable to locate model checkpoint")
        sys.exit(0)

    def compute_mutation_fitness_prediction(msa_data, decode_model, list_mutations_location, num_samples, batch_size=256):

        list_mutations=pd.read_csv(list_mutations_location, header=0)
        list_valid_mutations = ['wt']
        list_valid_mutated_sequences = {}
        list_valid_mutated_sequences['wt'] = msa_data.focus_seq_trimmed # first sequence in the list is the wild_type
        for mutation in list_mutations['mutations']:
            individual_substitutions = mutation.split(':')
            mutated_sequence = list(msa_data.focus_seq_trimmed)[:]
            fully_valid_mutation = True
            for mut in individual_substitutions:
                wt_aa, pos, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
                if pos not in msa_data.uniprot_focus_col_to_wt_aa_dict or msa_data.uniprot_focus_col_to_wt_aa_dict[pos] != wt_aa or mut not in msa_data.mutant_to_letter_pos_idx_focus_list:
                    print ("Not a valid mutant: "+mutation)
                    fully_valid_mutation = False
                    break
                else:
                    wt_aa,pos,idx_focus = msa_data.mutant_to_letter_pos_idx_focus_list[mut]
                    mutated_sequence[idx_focus] = mut_aa #perform the corresponding AA substitution
            
            if fully_valid_mutation:
                list_valid_mutations.append(mutation)
                list_valid_mutated_sequences[mutation] = ''.join(mutated_sequence)

        mutated_sequences_one_hot = np.zeros((len(list_valid_mutations),len(msa_data.focus_cols),len(msa_data.alphabet)))
        for i,mutation in enumerate(list_valid_mutations):
            sequence = list_valid_mutated_sequences[mutation]
            for j,letter in enumerate(sequence):
                if letter in msa_data.aa_dict:
                    k = msa_data.aa_dict[letter]
                    mutated_sequences_one_hot[i,j,k] = 1.0

        mutated_sequences_one_hot = torch.tensor(mutated_sequences_one_hot)
        dataloader = torch.utils.data.DataLoader(mutated_sequences_one_hot, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        prediction_matrix = torch.zeros((len(list_valid_mutations),num_samples))

        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(dataloader, 'Looping through mutation batches')):
                x = batch.type(torch.float32).to(device)
                for j in tqdm.tqdm(range(num_samples), 'Looping through number of samples for batch #: '+str(i+1)):
                    seq_predictions = decode_model.all_likelihood_components(x)
                    prediction_matrix[i*batch_size:i*batch_size+len(x),j] = seq_predictions
                tqdm.tqdm.write('\n')
            mean_predictions = prediction_matrix.mean(dim=1, keepdim=False)
            std_predictions = prediction_matrix.std(dim=1, keepdim=False)
            delta_elbos = mean_predictions - mean_predictions[0]
            mutation_fitness_prediction =  - delta_elbos.detach().cpu().numpy()

        return list_valid_mutations, mutation_fitness_prediction, mean_predictions[0].detach().cpu().numpy(), std_predictions.detach().cpu().numpy()
    
    list_valid_mutations, mutation_fitness_prediction, _, _ = compute_mutation_fitness_prediction(msa_data=data,
                                                                    decode_model=model,
                                                    list_mutations_location=args.mutations_location, 
                                                    num_samples=args.num_samples_compute_predictions,
                                                    batch_size=args.batch_size)

    df = {}
    df['protein_name'] = protein_name
    df['mutations'] = list_valid_mutations
    df['model_prediction'] = mutation_fitness_prediction
    df = pd.DataFrame(df)
    
    mutation_fitness_prediction_output_filename = args.output_prediction_location+os.sep+protein_name+'_'+str(args.num_samples_compute_predictions)+'_samples'+args.output_prediction_filename_suffix+'.csv'
    try:
        keep_header = os.stat(mutation_fitness_prediction_output_filename).st_size == 0
    except:
        keep_header=True 
    df.to_csv(path_or_buf=mutation_fitness_prediction_output_filename, index=False, mode='a', header=keep_header)
