import pandas as pd
from Bio import SeqIO
import numpy as np
import random


def calculate_strength_and_specificity(switches):
    # Compute the strength and specificity of each switch / Mean(Top3_x)
    feature_range = range(1, 890)
    top3_mean_values = switches.iloc[:, feature_range].apply(lambda row: row.nlargest(3).mean(), axis=1)
    all_sum_values = switches.iloc[:, feature_range].apply(lambda row: row.sum(), axis=1)
    top3_specificity = top3_mean_values / all_sum_values
    switches['Strength'] = top3_mean_values
    switches['Specificity'] = top3_specificity
    return switches

def Switch_evaluator_load():
    # Peak Load，184827
    bed_file_path = "public_dataset/hg19.cage_peak_tpm.osc.txt"
    bed_data = pd.read_csv(bed_file_path, sep='\t', header=None, skiprows=896)
    # print(bed_data)

    # Delete housekeeping genes( Median(x)<=0.2 ) / 145387
    row_medians = bed_data.iloc[:, 1:].median(axis=1)
    ubiquitous_genes = bed_data[row_medians <= 0.2]
    # print(ubiquitous_genes)

    # Delete broad promoters ( width(x)<11 ) / 49413
    coordinates = ubiquitous_genes.iloc[:, 0].str.extract(r':(\d+)\.\.(\d+)').astype(float)
    coordinate_diff = coordinates[1] - coordinates[0]
    sharp_genes = ubiquitous_genes[coordinate_diff < 11]
    # print(sharp_genes)

    # Delete low-quality peaks ( count(x)>=3 ) / 49235
    counts_file = pd.read_csv("public_dataset/hg19.cage_peak_counts.osc.txt", sep='\t', header=None, skiprows=893)
    row_maximum = counts_file.iloc[:, 1:].max(axis=1)
    switches = sharp_genes[row_maximum >= 10]
    # print(switches)

    # Calculate the strength and specificity of each switch
    switches = calculate_strength_and_specificity(switches)
    # print(switches)

    # Get the nucleotide sequences of core promoter
    hg19_file_path = 'public_dataset/hg19.fa'
    hg19_genome = SeqIO.to_dict(SeqIO.parse(hg19_file_path, 'fasta'))
    Sequences = []
    Chromosome = []
    Strands = []
    Strength = []
    Specificity = []
    for index, row in switches.iterrows():
        parts = row[0].split(':')
        chrom = parts[0]
        coords_strand = parts[1].split(',')
        coordinates = coords_strand[0].split('..')
        start = int(coordinates[0])
        end = int(coordinates[1])
        strand = coords_strand[1]
        mid_pos = int((start + end) / 2)
        if chrom != 'chrM':
            if strand == '+':
                sequence = hg19_genome[chrom].seq[mid_pos - 50:mid_pos + 10]
            elif strand == '-':
                sequence = hg19_genome[chrom].seq[mid_pos - 10:mid_pos + 50].reverse_complement()
            else:
                sequence = ''
                print('Error!')
            sequence = sequence.upper()
            Sequences.append(sequence)
            Chromosome.append(chrom)
            Strands.append(strand)
            Strength.append(row['Strength'])
            Specificity.append(row['Specificity'])

    # Switch dataset save
    df = pd.DataFrame({'Sequences': Sequences, 'Chromosome': Chromosome, 'Strands': Strands,
                       'Strength': Strength, 'Specificity': Specificity})
    df.to_csv('DeepCP_dataset/switch_60bp.csv', index=False)
    print(f"Data has been saved !")


def Data_distribution():
    # Compute the number of switches with top X%
    data = pd.read_csv('switch.csv')
    Strength = data['Strength'].tolist()
    # Specificity = data['Specificity'].tolist()
    test = Strength
    q = np.percentile(test, 70)
    last = [value for value in test if value >= q]
    print(q, len(last))


def CP_load():
    # Peak Load，184827
    bed_file_path = "public_dataset/hg19.cage_peak_tpm.osc.txt"
    bed_data = pd.read_csv(bed_file_path, sep='\t', header=None, skiprows=896)
    # print(bed_data)

    # Define the range of each chromosome and genome
    hg19_file_path = 'public_dataset/hg19.fa'
    hg19_genome = SeqIO.to_dict(SeqIO.parse(hg19_file_path, 'fasta'))
    Range_pos = {'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260,
                 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022, 'chr9': 141213431, 'chr10': 135534747,
                 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392,
                 'chr16': 90354753, 'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr20': 63025520,
                 'chr21': 48129895, 'chr22': 51304566, 'chrX': 155270560, 'chrY': 59373566}

    Sequences = []
    Chromosome = []
    Strands = []
    Label = []
    # Obtain natural CP sequences and identify peak positions
    Position_by_chrom = {}
    for index, row in bed_data.iterrows():
        parts = row[0].split(':')
        chrom = parts[0]
        coords_strand = parts[1].split(',')
        coordinates = coords_strand[0].split('..')
        start = int(coordinates[0])
        end = int(coordinates[1])
        strand = coords_strand[1]
        mid_pos = int((start + end) / 2)
        if chrom != 'chrM':
            # Obtain the sequence of the peak
            if strand == '+':
                sequence = hg19_genome[chrom].seq[mid_pos - 50:mid_pos + 10]
            elif strand == '-':
                sequence = hg19_genome[chrom].seq[mid_pos - 10:mid_pos + 50].reverse_complement()
            else:
                sequence = ''
                print('Error!')

            if 'N' not in sequence:
                # Obtain the position of the peak
                if chrom not in Position_by_chrom.keys():
                    Position_by_chrom[chrom] = []
                Position_by_chrom[chrom].append(mid_pos)

                sequence = sequence.upper()
                Sequences.append(sequence)
                Chromosome.append(chrom)
                Strands.append(strand)
                Label.append(1)

    # Generate random sequences (ensure the number of samples in each chromosome is equal)
    for key in Position_by_chrom.keys():
        existing_pos = np.array(Position_by_chrom[key])
        new_pos = []
        while len(new_pos) < len(existing_pos):
            # Generate random sequences
            new_value = random.randint(1 + 50, Range_pos[key] - 50)
            new_strand = random.choice(["+", "-"])
            if new_strand == '+':
                sequence = hg19_genome[key].seq[new_value - 50:new_value + 10]
            elif new_strand == '-':
                sequence = hg19_genome[key].seq[new_value - 10:new_value + 50].reverse_complement()
            else:
                sequence = ''
                print('Position Error!')

            # Evaluate if the criteria are satisfied
            if np.all(np.abs(new_value - existing_pos) > 100) and 'N' not in sequence:
                new_pos.append(new_value)
                sequence = sequence.upper()
                Sequences.append(sequence)
                Chromosome.append(key)
                Strands.append(new_strand)
                Label.append(0)

    df = pd.DataFrame({'Sequences': Sequences, 'Chromosome': Chromosome, 'Strands': Strands, 'Label': Label})
    df.to_csv('DeepCP_dataset/CP_discriminator.csv', index=False)
    print(f"Data has been saved !")


def Switch_discriminator_load():
    # Peak Load，184827
    bed_file_path = "public_dataset/hg19.cage_peak_tpm.osc.txt"
    bed_data = pd.read_csv(bed_file_path, sep='\t', header=None, skiprows=896)
    # print(bed_data)

    # Delete housekeeping genes( Median(x)<=0.2 ) / 145387
    row_medians = bed_data.iloc[:, 1:].median(axis=1)
    ubiquitous_genes = bed_data[row_medians <= 0.2]
    # print(ubiquitous_genes)

    # Delete broad promoters ( width(x)<11 ) / 49413
    coordinates = ubiquitous_genes.iloc[:, 0].str.extract(r':(\d+)\.\.(\d+)').astype(float)
    coordinate_diff = coordinates[1] - coordinates[0]
    sharp_genes = ubiquitous_genes[coordinate_diff < 11]
    # print(sharp_genes)

    # Delete low-quality peaks ( count(x)>=3 ) / 49235
    counts_file = pd.read_csv("public_dataset/hg19.cage_peak_counts.osc.txt", sep='\t', header=None, skiprows=893)
    row_maximum = counts_file.iloc[:, 1:].max(axis=1)
    switches = sharp_genes[row_maximum >= 10]
    # print(switches)

    # Define switch or not
    bed_data['SampleType'] = 0
    bed_data.loc[switches.index, 'SampleType'] = 1
    positive_samples = bed_data[bed_data['SampleType'] == 1]
    negative_samples = bed_data[bed_data['SampleType'] == 0]
    # print(positive_samples)
    # print(negative_samples)

    # Obtain the nucleotide sequences of switch
    hg19_file_path = 'public_dataset/hg19.fa'
    hg19_genome = SeqIO.to_dict(SeqIO.parse(hg19_file_path, 'fasta'))
    Sequences = []
    Chromosome = []
    Strands = []
    Label = []
    Positive_by_chrom = {}
    for index, row in positive_samples.iterrows():
        parts = row[0].split(':')
        chrom = parts[0]
        coords_strand = parts[1].split(',')
        coordinates = coords_strand[0].split('..')
        start = int(coordinates[0])
        end = int(coordinates[1])
        strand = coords_strand[1]
        mid_pos = int((start + end) / 2)
        if chrom != 'chrM':
            # Obtain the sequence of the peak
            if strand == '+':
                sequence = hg19_genome[chrom].seq[mid_pos - 50:mid_pos + 10]
            elif strand == '-':
                sequence = hg19_genome[chrom].seq[mid_pos - 10:mid_pos + 50].reverse_complement()
            else:
                sequence = ''
                print('Error!')

            if 'N' not in sequence:
                # Obtain the position of the peak
                if chrom not in Positive_by_chrom.keys():
                    Positive_by_chrom[chrom] = 0
                Positive_by_chrom[chrom] += 1

                sequence = sequence.upper()
                Sequences.append(sequence)
                Chromosome.append(chrom)
                Strands.append(strand)
                Label.append(1)

    # Calculate the number of samples of each chromosome in negative samples
    Negative_by_chrom = {}
    for index, row in negative_samples.iterrows():
        parts = row[0].split(':')
        chrom = parts[0]
        coords_strand = parts[1].split(',')
        coordinates = coords_strand[0].split('..')
        start = int(coordinates[0])
        end = int(coordinates[1])
        strand = coords_strand[1]
        mid_pos = int((start + end) / 2)
        if chrom != 'chrM':
            # Obtain the sequence of the peak
            if strand == '+':
                sequence = hg19_genome[chrom].seq[mid_pos - 50:mid_pos + 10]
            elif strand == '-':
                sequence = hg19_genome[chrom].seq[mid_pos - 10:mid_pos + 50].reverse_complement()
            else:
                sequence = ''
                print('Error!')
            if 'N' not in sequence:
                if chrom not in Negative_by_chrom.keys():
                    Negative_by_chrom[chrom] = 0
                Negative_by_chrom[chrom] += 1

    Count_by_chrom = {}
    # Obtain the nucleotide sequences of non-switch
    for index, row in negative_samples.iterrows():
        parts = row[0].split(':')
        chrom = parts[0]
        coords_strand = parts[1].split(',')
        coordinates = coords_strand[0].split('..')
        start = int(coordinates[0])
        end = int(coordinates[1])
        strand = coords_strand[1]
        mid_pos = int((start + end) / 2)
        if chrom != 'chrM':
            # Obtain the sequence of the peak
            if strand == '+':
                sequence = hg19_genome[chrom].seq[mid_pos - 50:mid_pos + 10]
            elif strand == '-':
                sequence = hg19_genome[chrom].seq[mid_pos - 10:mid_pos + 50].reverse_complement()
            else:
                sequence = ''
                print('Error!')
            if 'N' not in sequence:
                if chrom not in Count_by_chrom.keys():
                    Count_by_chrom[chrom] = 0
                Count_by_chrom[chrom] += 1

                if Count_by_chrom[chrom] <= min(Positive_by_chrom[chrom], Negative_by_chrom[chrom]):
                    sequence = sequence.upper()
                    Sequences.append(sequence)
                    Chromosome.append(chrom)
                    Strands.append(strand)
                    Label.append(0)

    # Switch or not dataset save
    df = pd.DataFrame({'Sequences': Sequences, 'Chromosome': Chromosome,
                       'Strands': Strands, 'Label': Label})
    df.to_csv('DeepCP_dataset/switch_or_not.csv', index=False)
    print(f"Data has been saved !")


def Housekeeping_developmental_load():
    # Peak Load，184827
    bed_file_path = "public_dataset/hg19.cage_peak_tpm.osc.txt"
    bed_data = pd.read_csv(bed_file_path, sep='\t', header=None, skiprows=896)
    print(bed_data)

    # Remain housekeeping genes( Median(x)>0.2 ) / 39440
    row_medians = bed_data.iloc[:, 1:].median(axis=1)
    ubiquitous_genes = bed_data[row_medians > 0.2]
    print(ubiquitous_genes)

    # Remain broad promoters ( width(x)>=11 ) / 36111
    coordinates = ubiquitous_genes.iloc[:, 0].str.extract(r':(\d+)\.\.(\d+)').astype(float)
    coordinate_diff = coordinates[1] - coordinates[0]
    sharp_genes = ubiquitous_genes[coordinate_diff >= 11]
    print(sharp_genes)

    # Delete low-quality peaks ( count(x)>=3 ) / 36111
    counts_file = pd.read_csv("public_dataset/hg19.cage_peak_counts.osc.txt", sep='\t', header=None, skiprows=893)
    row_maximum = counts_file.iloc[:, 1:].max(axis=1)
    switches = sharp_genes[row_maximum >= 10]
    print(switches)

    # Obtain the nucleotide sequences of switch
    hg19_file_path = 'public_dataset/hg19.fa'
    hg19_genome = SeqIO.to_dict(SeqIO.parse(hg19_file_path, 'fasta'))
    Sequences = []
    Chromosome = []
    Strands = []
    for index, row in switches.iterrows():
        parts = row[0].split(':')
        chrom = parts[0]
        coords_strand = parts[1].split(',')
        coordinates = coords_strand[0].split('..')
        start = int(coordinates[0])
        end = int(coordinates[1])
        strand = coords_strand[1]
        mid_pos = int((start + end) / 2)
        if chrom != 'chrM':
            # Obtain the sequence of the peak
            if strand == '+':
                sequence = hg19_genome[chrom].seq[mid_pos - 50:mid_pos + 10]
            elif strand == '-':
                sequence = hg19_genome[chrom].seq[mid_pos - 10:mid_pos + 50].reverse_complement()
            else:
                sequence = ''
                print('Error!')

            if 'N' not in sequence:
                sequence = sequence.upper()
                Sequences.append(sequence)
                Chromosome.append(chrom)
                Strands.append(strand)

    # Switch or not dataset save
    print(len(Sequences)) #35816
    df = pd.DataFrame({'Sequences': Sequences, 'Chromosome': Chromosome,
                       'Strands': Strands})
    df.to_csv('DeepCP_dataset/Housekeeping_genes.csv', index=False)
    print(f"Data has been saved !")


if __name__ == '__main__':
    # Switch_evaluator_load()
    # Data_distribution()
    # CP_load()
    # Switch_discriminator_load()
    Housekeeping_developmental_load()
