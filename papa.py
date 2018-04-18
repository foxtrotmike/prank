
"""
PAPA - a program for predicting the prion propensity of a protein

FoldIndex formula:
2.785(H) - |R| - 1.151
H = sum of the hydrophobicities across the window							
R = net charge (where D/E=-1; K/R=+1; all others = neutral (including H))							
"""

#window_size = 41
#half_window_size = int(window_size / 2)

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

propensities = {'A' :-0.396490246, 'C' : 0.415164505, 'D' : -1.276997939, 'E' : -0.605023827, 'F' : 0.838732498,
                'G' : -0.039220713, 'H': -0.278573356, 'I' : 0.813697862, 'K': -1.576748587, 'L' : -0.040005335,
                'M' : 0.673729095, 'N' : 0.080295334, 'P' : -1.197447496, 'Q' : 0.069168387, 'R' : -0.405858577,
                'S' : 0.133912418, 'T' : -0.11457038, 'V' : 0.813697862, 'W' : 0.666735081, 'Y' : 0.77865336}
																			
hydrophobicity = {'A' : 0.7, 'C' : 0.777777778, 'D' : 0.111111111, 'E' : 0.111111111, 'F' : 0.811111111,
                  'G' : 0.455555556, 'H' : 0.144444444, 'I' : 1, 'K' : 0.066666667, 'L': 0.922222222,
                  'M' : 0.711111111, 'N' : 0.111111111, 'P' : 0.322222222, 'Q' : 0.111111111, 'R' : 0,
                  'S' : 0.411111111, 'T' : 0.422222222, 'V' : 0.966666667, 'W' : 0.4, 'Y' : 0.355555556}

charge = {'A' : 0, 'C' : 0, 'D' : -1, 'E' : -1, 'F' : 0,
          'G' : 0, 'H' : 0, 'I' : 0, 'K' : 1, 'L': 0,
          'M' : 0, 'N' : 0, 'P' : 0, 'Q' : 0, 'R' : 1,
          'S' : 0, 'T' : 0, 'V' : 0, 'W' : 0, 'Y' : 0}


def window_scores(args, sequence, aa_dict, ignore_consecutive_prolines = False) :

    return [window_score(args, sequence, i, aa_dict, ignore_consecutive_prolines) for i in range(len(sequence))]

def window_score(args, sequence, position, aa_dict, ignore_consecutive_prolines = False) :
    start,end = get_window(args, sequence, position)
    score = 0.0
    for i in range(start, end) :
        if sequence[i] not in amino_acids :
            continue
        if (not ignore_consecutive_prolines) :
            score += aa_dict[sequence[i]]
        else :
            if (sequence[i] != 'P') :
                score += aa_dict[sequence[i]]
            elif ((i > 0 and sequence[i-1] == 'P') or (i > 1 and sequence[i-2] == 'P')) :
                pass
            else :
                score += aa_dict[sequence[i]]
    return score / (end - start)

def get_window(args, sequence, position) :
    start = max(position - args.half_window_size, 0)
    end = min(len(sequence), position + args.half_window_size + 1)
    return start,end
                                   

def fold_index(args, sequence) :
    charges = window_scores(args, sequence, charge)
    hydrophobicities = window_scores(args, sequence, hydrophobicity)
    fold_index_list = [2.785 * hydrophobicities[i]  - abs(charges[i]) - 1.151 for i in range(len(sequence))]
    return super_window_scores(args, sequence, fold_index_list)


def super_window_scores(args, sequence, window_scores, fold_index_scores = None) :
    scores = []
    for i in range(len(sequence) - args.window_size) :
        if (fold_index_scores is not None and fold_index_scores[i] > 0) :
            scores.append(None)
        else :
            score = 0.0
            weights = 0.0
            for j in range(i, i + args.window_size) :
                start,end = get_window(args, sequence, j)
                score += (end - start) * window_scores[j]
                weights += (end - start)
            scores.append(score / weights)
    return scores
    

def classify(args, sequence, ignore_fold_index = False) :

    fold_index_list = fold_index(args, sequence)
    #print 'number of positions with negative fold index', sum([1 for f in fold_index_list if f < 0]), 'out of', len(sequence)

    window_propensities = window_scores(args, sequence, propensities, True)    
    if ignore_fold_index :
        scores = super_window_scores(args, sequence, window_propensities)
    else :
        scores = super_window_scores(args, sequence, window_propensities, fold_index_list)
    max_score = max(scores)
    max_position = scores.index(max_score)+args.half_window_size+1
    
    # the case when no window had a negative foldIndex    
    if max_score is None :
        max_score = -1.0
        max_position = -1

    return max_score, max_position, scores, fold_index_list

class MalformedInput :
    "Exception raised when the input file does not look like a fasta file."
    pass

class FastaRecord :
    "a fasta record."
    def __init__(self, header, sequence):
        self.header = header
        self.sequence = sequence

def _fasta_itr(file_handle) :
    "Provide an iteration through the fasta records in file."

    h = file_handle.readline()[:-1]
    if h[0] != '>':
        raise MalformedInput()
    h = h[1:]

    seq = []
    for line in file_handle:
        line = line[:-1] # remove newline
        if len(line)==0:
            continue
        if line[0] == '>':
            yield FastaRecord(h,''.join(seq))
            h = line[1:]
            seq = []
            continue
        seq.append(line)
    yield FastaRecord(h,''.join(seq))

class fasta_itr (object) :
    "An iterator through a sequence of fasta records."
    def __init__(self, src) :
        self.__itr = _fasta_itr(src)

    def __iter__(self) :
        return self

    def next(self) :
        return self.__itr.next()


def run(args) :

    if args.outfile is None :
        outfile = sys.stdout
    else :
        outfile = open(args.outfile, 'w')
    for fasta_record in fasta_itr(open(args.fasta_input)) :
        sequence_id = fasta_record.header
        sequence = fasta_record.sequence.upper()
        if len(sequence) <= args.window_size :
            outfile.write(str(sequence_id) + ',' + 'protein length below window size\n')
            continue
        score,pos,scores,fold_indexes = classify(args, sequence, args.ignore_fold_index)
        if args.verbose :
            scores_token = '[' + ' '.join([str(s) for s in scores ]) + ']'
            fold_index_token = '[' + ' '.join([str(f) for f in fold_indexes ]) + ']'
            outfile.write(str(sequence_id) + ',' + str(score) + ',' + str(pos) +',' +
                          scores_token + ',' + fold_index_token + '\n')
        else :
            outfile.write(str(sequence_id) + ',' + str(score) + ',' + str(pos) + '\n')
    
def test() :
    scores = {'Ure2' : 0.1031, 'Sup35': 0.0997, 'Rnq1' : 0.1413, 'New1' : 0.1398, 'Nsp1' : 0.0239, 'Puf2' : 0.0768, 'Pub1' : 0.1551}
    sequences = {}
    for fasta_record in fasta_itr(open('sequences.fasta')) :
        sequences[fasta_record.header] = fasta_record.sequence
    for id in sequences :
        score,pos,scores,fold_index = classify(sequences[id])
        print id, len(sequences[id]), pos, score, scores[id], score / scores[id]
    print 'scores ignoring foldIndex'
    for id in sequences :
        score,pos,scores,fold_index = classify(sequences[id], True)
        print id, len(sequences[id]), pos, score, scores[id], score / scores[id]

def parse_arguments(arguments) :
    import argparse
    parser = argparse.ArgumentParser(description='Predict whether a given protein is prion forming', prog = 'papa')
    parser.add_argument('fasta_input', help = 'the input file (in fasta format)')
    parser.add_argument('-o', '--outfile', type = str,
                        help = """the output file. In non-verbose mode the output is a comma delimited file the columns are:
                        sequence id, maximum score, position
                        maximum score is the largest window score, and position is the position where it occurs.
                        In verbose mode the output also includes the scores for the whole protein, and the fold index scores.
                        If not output file is given, results are printed to the screen.
                        """)
    parser.add_argument('-v', '--verbose', action='store_true', help = 'in verbose mode the output includes the scores for every position as well as the per-position fold index scores')
    parser.add_argument('--ignore_fold_index', action='store_true',
                        help = """Whether to ignore the fold index values for the protein (default: False)
                        Set it to True to ignore fold index.
                        """)
    parser.add_argument('--window_size', type = int, default = 41,
                        help = """The window size used by PAPA (default = 41).
                        Proteins that are shorter than the window size are not classified.
                        """)
    args = parser.parse_args(arguments)
    args.half_window_size = int(args.window_size / 2)    
    return args

if __name__ == '__main__' :

    import sys
    args = parse_arguments(sys.argv[1:])
    run(args)
    #test()
