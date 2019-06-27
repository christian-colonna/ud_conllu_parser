import pyconll as pcl
import io
import matplotlib.pyplot as plt
import argparse
import sys
from conll18_ud_eval import *
from collections import defaultdict
from copy import copy



FUNCTIONAL_DEPRELS = {
    "aux", "cop", "mark", "det", "clf", "case", "cc", "nsubj", "obj", "iobj", 
    "csubj", "ccomp", "xcomp", "obl", "vocative", "expl", "dislocated", "advcl", "advmod", "discourse", "nmod", "appos",
    "nummod", "acl", "amod", "conj", "fixed", "flat", "compound", "list",
    "parataxis", "orphan", "goeswith", "reparandum", "root", "dep", "punct"
}

def _decode(text):
    return text if sys.version_info[0] >= 3 or not isinstance(text, str) else text.decode("utf-8")

def _encode(text):
    return text if sys.version_info[0] >= 3 or not isinstance(text, unicode) else text.encode("utf-8")
    
def load_conllu(file):
    # Internal representation classes
    class UDRepresentation:
        def __init__(self):
            # Characters of all the tokens in the whole file.
            # Whitespace between tokens is not included.
            self.characters = []
            # List of UDSpan instances with start&end indices into `characters`.
            self.tokens = []
            # List of UDWord instances.
            self.words = []
            # List of UDSpan instances with start&end indices into `characters`.
            self.sentences = []
    class UDSpan:				# i tokens
        def __init__(self, start, end):
            self.start = start
            # Note that self.end marks the first position **after the end** of span,
            # so we can use characters[start:end] or range(start, end).
            self.end = end
    class UDWord:		# c'è ereditarietà. Lo span di word è un oggetto. se faccio .start 
        def __init__(self, span, columns, is_multiword):
            # Span of this word (or MWT, see below) within ud_representation.characters.
            self.span = span
            # 10 columns of the CoNLL-U file: ID, FORM, LEMMA,...
            self.columns = columns
            # is_multiword==True means that this word is part of a multi-word token.
            # In that case, self.span marks the span of the whole multi-word token.
            self.is_multiword = is_multiword
            # Reference to the UDWord instance representing the HEAD (or None if root).
            self.parent = None
            # List of references to UDWord instances representing functional-deprel children.
            self.functional_children = []
            # Only consider universal FEATS.
            self.columns[FEATS] = "|".join(sorted(feat for feat in columns[FEATS].split("|")
                                                  if feat.split("=", 1)[0] in UNIVERSAL_FEATURES))
            # Let's ignore language-specific deprel subtypes.
            self.columns[DEPREL] = columns[DEPREL].split(":")[0]
            # Precompute which deprels are CONTENT_DEPRELS and which FUNCTIONAL_DEPRELS
            self.is_content_deprel = self.columns[DEPREL] in CONTENT_DEPRELS
            self.is_functional_deprel = self.columns[DEPREL] in FUNCTIONAL_DEPRELS

    ud = UDRepresentation()

    # Load the CoNLL-U file
    index, sentence_start = 0, None
    while True:
        line = file.readline()
        if not line:
            break
        line = _decode(line.rstrip("\r\n"))

        # Handle sentence start boundaries
        if sentence_start is None:
            # Skip comments
            if line.startswith("#"):
                continue
            # Start a new sentence
            ud.sentences.append(UDSpan(index, 0))
            sentence_start = len(ud.words)
        if not line:
            # Add parent and children UDWord links and check there are no cycles
            def process_word(word):
                if word.parent == "remapping":
                    raise UDError("There is a cycle in a sentence")
                if word.parent is None:
                    head = int(word.columns[HEAD])
                    if head < 0 or head > len(ud.words) - sentence_start:
                        raise UDError("HEAD '{}' points outside of the sentence".format(_encode(word.columns[HEAD])))
                    if head:
                        parent = ud.words[sentence_start + head - 1]
                        word.parent = "remapping"
                        process_word(parent)
                        word.parent = parent

            for word in ud.words[sentence_start:]:
                process_word(word)
            # func_children cannot be assigned within process_word
            # because it is called recursively and may result in adding one child twice.
            for word in ud.words[sentence_start:]:
                if word.parent and word.is_functional_deprel:
                    word.parent.functional_children.append(word)

            # Check there is a single root node
            if len([word for word in ud.words[sentence_start:] if word.parent is None]) != 1:
                raise UDError("There are multiple roots in a sentence")

            # End the sentence
            ud.sentences[-1].end = index
            sentence_start = None
            continue

        # Read next token/word
        columns = line.split("\t")
        if len(columns) != 10:
            raise UDError("The CoNLL-U line does not contain 10 tab-separated columns: '{}'".format(_encode(line)))

        # Skip empty nodes
        if "." in columns[ID]:
            continue

        # Delete spaces from FORM, so gold.characters == system.characters
        # even if one of them tokenizes the space. Use any Unicode character
        # with category Zs.
        columns[FORM] = "".join(filter(lambda c: unicodedata.category(c) != "Zs", columns[FORM]))
        if not columns[FORM]:
            raise UDError("There is an empty FORM in the CoNLL-U file")

        # Save token
        ud.characters.extend(columns[FORM])
        ud.tokens.append(UDSpan(index, index + len(columns[FORM])))
        index += len(columns[FORM])

        # Handle multi-word tokens to save word(s)
        if "-" in columns[ID]:
            try:
                start, end = map(int, columns[ID].split("-"))
            except:
                raise UDError("Cannot parse multi-word token ID '{}'".format(_encode(columns[ID])))

            for _ in range(start, end + 1):
                word_line = _decode(file.readline().rstrip("\r\n"))
                word_columns = word_line.split("\t")
                if len(word_columns) != 10:
                    raise UDError("The CoNLL-U line does not contain 10 tab-separated columns: '{}'".format(_encode(word_line)))
                ud.words.append(UDWord(ud.tokens[-1], word_columns, is_multiword=True))
        # Basic tokens/words
        else:
            try:
                word_id = int(columns[ID])
            except:
                raise UDError("Cannot parse word ID '{}'".format(_encode(columns[ID])))
            if word_id != len(ud.words) - sentence_start + 1:
                raise UDError("Incorrect word ID '{}' for word '{}', expected '{}'".format(
                    _encode(columns[ID]), _encode(columns[FORM]), len(ud.words) - sentence_start + 1))

            try:
                head_id = int(columns[HEAD])
            except:
                raise UDError("Cannot parse HEAD '{}'".format(_encode(columns[HEAD])))
            if head_id < 0:
                raise UDError("HEAD cannot be negative")

            ud.words.append(UDWord(ud.tokens[-1], columns, is_multiword=False))

    if sentence_start is not None:
        raise UDError("The CoNLL-U file does not end with empty line")

    return ud

def load_conllu_file(path):
    _file = open(path, mode="r", **({"encoding": "utf-8"} if sys.version_info >= (3, 0) else {}))
    return load_conllu(_file)

class Score:
    def __init__(self, gold_total, system_total, correct, aligned_total=None):
        self.correct = correct
        self.gold_total = gold_total
        self.system_total = system_total
        self.aligned_total = aligned_total
        self.precision = correct / system_total if system_total else 0.0
        self.recall = correct / gold_total if gold_total else 0.0
        self.f1 = 2 * correct / (system_total + gold_total) if system_total + gold_total else 0.0
        self.aligned_accuracy = correct / aligned_total if aligned_total else aligned_total
class AlignmentWord:
    def __init__(self, gold_word, system_word):
        self.gold_word = gold_word
        self.system_word = system_word
class Alignment:
    def __init__(self, gold_words, system_words):
        self.gold_words = gold_words
        self.system_words = system_words
        self.matched_words = []
        self.matched_words_map = {}
    def append_aligned_words(self, gold_word, system_word):
        self.matched_words.append(AlignmentWord(gold_word, system_word))
        self.matched_words_map[system_word] = gold_word
			

def alignment_score(alignment, key_fn=None, filter_fn=None, debug=False):
    if filter_fn is not None:
        gold = sum(1 for gold in alignment.gold_words if filter_fn(gold))
        system = sum(1 for system in alignment.system_words if filter_fn(system))
        aligned = sum(1 for word in alignment.matched_words if filter_fn(word.gold_word))
    else:
        gold = len(alignment.gold_words)
        system = len(alignment.system_words)
        aligned = len(alignment.matched_words)

    if key_fn is None:
        # Return score for whole aligned words
        return Score(gold, system, aligned)

    def gold_aligned_gold(word):
        return word
    def gold_aligned_system(word):
        return alignment.matched_words_map.get(word, "NotAligned") if word is not None else None
    correct = 0
    for words in alignment.matched_words:
        if filter_fn is None or filter_fn(words.gold_word):
            if key_fn(words.gold_word, gold_aligned_gold) == key_fn(words.system_word, gold_aligned_system):
                correct += 1
            elif debug:
                print( key_fn(words.gold_word, gold_aligned_gold), "!=", key_fn(words.system_word, gold_aligned_system))

    return Score(gold, system, correct, aligned)


def align_words(gold_words, system_words):
    alignment = Alignment(gold_words, system_words)

    gi, si = 0, 0
    while gi < len(gold_words) and si < len(system_words):
        if gold_words[gi].is_multiword or system_words[si].is_multiword:
            # A: Multi-word tokens => align via LCS within the whole "multiword span".
            gs, ss, gi, si = find_multiword_span(gold_words, system_words, gi, si)

            if si > ss and gi > gs:
                lcs = compute_lcs(gold_words, system_words, gi, si, gs, ss)

                # Store aligned words
                s, g = 0, 0
                while g < gi - gs and s < si - ss:
                    if gold_words[gs + g].columns[FORM].lower() == system_words[ss + s].columns[FORM].lower():
                        alignment.append_aligned_words(gold_words[gs+g], system_words[ss+s])
                        g += 1
                        s += 1
                    elif lcs[g][s] == (lcs[g+1][s] if g+1 < gi-gs else 0):
                        g += 1
                    else:
                        s += 1
        else:
            # B: No multi-word token => align according to spans.
            if (gold_words[gi].span.start, gold_words[gi].span.end) == (system_words[si].span.start, system_words[si].span.end):
                alignment.append_aligned_words(gold_words[gi], system_words[si])
                gi += 1
                si += 1
            elif gold_words[gi].span.start <= system_words[si].span.start:
                gi += 1
            else:
                si += 1

    return alignment
    

def get_dep_distances(ds_ud_words):
	""" La funzione processa processa le parole del ds_ud (ds_ud.words) e ritorna un dizionario con 
		 il numero di occorrenze delle distanze tra parole dipendenti come valore e la distanza tra parole dipendenti come chiave.
		 La distanza di una dipendenza tra due parole wi, wj è data da |i - j|  
	"""
	conteggio = defaultdict(int)
	for w in ds_ud_words:
		dgold = abs(int(w.columns[ID]) - int(w.parent.columns[ID])) if w.parent != None else 0	
		if dgold != 0: conteggio[dgold] += 1
	return conteggio

def udword_distance_to_root(ud_word):
	""" La funzione calcola la distanza (int) di una parola ds_ud dalla root della frase"""
	if ud_word.parent != None: # ud_word non è root	--> posso calcolare d
		a = copy(ud_word.parent)
		while a.parent != None:
			a = copy(a.parent) # ud_word non dipende da root allora risalgo il grafo finchè $a non è root
		d = abs(int(ud_word.columns[ID]) - int(a.columns[ID]))
	else: d = 0
	return d		  

def dependency_distance(ud_word):
	return (abs(int(ud_word.columns[ID]) - int(ud_word.parent.columns[ID])) if ud_word.parent != None else 0)

def word_grade(ud_word):
	""" La funzione prende una parola ud e restituisce il numero di archi uscenti dalla dipendenza. La radice del grafo non ha dipendenze restituisco -1.
		 In teoria dei grafi è equivalente al grado del nodo. 
	"""
	return len(ud_word.parent.functional_children) if ud_word.parent is not None else -1	


def non_projective_grade(ds_ud):
	""" La funzione inietta un identificatore unico ad ogni parola del dataset e il grado di non proiettività relativo ad ogni arco.
		 Il grado di non proiettività di un arco di dipendenza da una parola w ad una parola w
		 è definito come il numero di parole che occorrono tra w ed u che non discendono da w e 
		 modificano una parola che non occorre tra w ed u.
	"""
	
	# inietto identificatore unico e inizializzo grado di non proiettività
	ID = 0
	for w in ds_ud.words:
		w.uniqueID = ID
		w.non_projectivity = 0		
		ID +=1

	for ud_word in ds_ud.words:
		# controllo che la parola non sia root
		if ud_word.parent is not None: 	
		# controllo se la parola modificata occorre prima o dopo il modificatore
			if ud_word.uniqueID < ud_word.parent.uniqueID:
				for w in ds_ud.words[ud_word.uniqueID+1:ud_word.parent.uniqueID]:
					if w.parent is not None:			
						if (w.parent.uniqueID < ud_word.uniqueID) or (w.parent.uniqueID > ud_word.parent.uniqueID):
							ud_word.non_projectivity += 1
			if ud_word.uniqueID > ud_word.parent.uniqueID:
				for w in ds_ud.words[ud_word.parent.uniqueID+1:ud_word.uniqueID]:
					if w.parent is not None:
						if (w.parent.uniqueID > ud_word.uniqueID) or (w.parent.uniqueID < ud_word.parent.uniqueID):
							ud_word.non_projectivity += 1

def get_modifier(ud_word):
	""" La funzione controlla se una parola è modificatrice di un'altra parola """
	
def evaluation_print(metric, parametro):
		formato = "{}{} |{:10} |{:10} |{:10} |{:10}" if parametro < 10 else "{}{}|{:10} |{:10} |{:10} |{:10}"
		formato2 = "{}{} |{:10.2f} |{:10.2f} |{:10.2f} |{}" if parametro < 10 else "{}{}|{:10.2f} |{:10.2f} |{:10.2f} |{}"
		print("Metric   | Correct   |      Gold | Predicted | Aligned")		
		print(formato.format(
			metric,
        	parametro,
        	evaluation[metric].correct,
        	evaluation[metric].gold_total,
        	evaluation[metric].system_total,
        	evaluation[metric].aligned_total or (evaluation[metric].correct if metric == "Words" else "") 
    	))
		print("Metric   | Precision |    Recall |  F1 Score | AligndAcc")
		print(formato2.format(
			metric,
        	parametro,
        	100 * evaluation[metric].precision,
        	100 * evaluation[metric].recall,
        	100 * evaluation[metric].f1,
        	"{:10.2f}\n".format(100 * evaluation[metric].aligned_accuracy) if evaluation[metric].aligned_accuracy is not None else ""))

def pos_deprel_evaluation_print(metric, parametro):
		
		formato = "{:7}|{}|{:10} |{:10} |{:10} |{:10}"
		formato2 = "{:7}|{}|{:10.2f} |{:10.2f} |{:10.2f} |{}"
		while len(parametro) != 14:
			if len(parametro) < 14:	parametro += ' '
			else: parametro -= ' '
		print("Metric | POS o DEPREL | Correct   |      Gold | Predicted | Aligned")		
		print(formato.format(
			metric,
        	parametro,
        	evaluation[metric].correct,
        	evaluation[metric].gold_total,
        	evaluation[metric].system_total,
        	evaluation[metric].aligned_total or (evaluation[metric].correct if metric == "Words" else "") 
    	))
		print("Metric | POS o DEPREL | Precision |    Recall |  F1 Score | AligndAcc")
		print(formato2.format(
			metric,
        	parametro,
        	100 * evaluation[metric].precision,
        	100 * evaluation[metric].recall,
        	100 * evaluation[metric].f1,
        	"{:10.2f}\n".format(100 * evaluation[metric].aligned_accuracy) if evaluation[metric].aligned_accuracy is not None else ""))	

def precision_recall_graph(x, y1, y2, xlabel, ylabel, metrica):
	fig, ax = plt.subplots()
	ax.set_ylabel("%s (%s)" %(ylabel, metrica))
	ax.set_xlabel(xlabel)	
	ax.plot(x, y1, label="PRECISION")
	ax.plot(x, y2, label="RECALL" ) 
	
	ax.grid()
	ax.legend()
	plt.show()

def pos_deprel_bar_graph(x_values, y_values, y_label, x_label):
	
	plt.bar(x_values, y_values, color="gbyrc")
	plt.ylabel(y_label)
	plt.xlabel(x_label)
	plt.show()

def pos_dep_occurences(ds_words, dep=False, pos_t=False, min_occur=25):
	""" La funzione conteggia le occorrenze di parti del discorso di un dataset ud e
		 le occorrenze di archi di dipendenza di un dataset ud. Con parametro dep o pos
		 stampa il grafico, default dep=True. Ritorna pos e dep.
		 Il parametro min_occur screma le relazioni di dipendenza con poche occorrenze al plotting
		 del grafico per renderlo più leggibile. 
	"""
	pos = defaultdict(int)
	deprel = defaultdict(int)
	for w in ds_words:
		pos[w.columns[4]] += 1
		deprel[w.columns[7]] += 1
	if dep: print(deprel)
	if pos_t: print(pos)
	#grafici
	if dep: plt.bar([i for i in deprel.keys() if deprel[i] > min_occur], [i for i in deprel.values() if i > min_occur], color = 'gbyrk')
	if pos_t: plt.bar(list(pos.keys()), pos.values(), color = 'gbyrk')
	plt.show() 
	return pos, dep

def ClearListe(*argv):
	for arg in argv: arg.clear()
	    
if __name__ == '__main__':
	
	# Parse arguments	
	parser = argparse.ArgumentParser()
	parser.add_argument("--maxdistance", "-d", type=int,
                        help="Max distance of dependencies beetwen words or from a word and sentence root on which compute statistics such as dep or root")
	parser.add_argument("--maxgrade", "-g", type=int,
                        help="Max number of modifier siblings of a word, or max number of non projective grade. Declare this to compute child statistics or non-projectivity")
	parser.add_argument("--UAS", "-U", default=False, action="store_true",
                        help="UAS metric, default both UAS and LAS, if you explicitly declare this parameter then only this metric will be evaluated")
	parser.add_argument("--LAS", "-L", default=False, action="store_true",
		                   help="LAS metric, default both UAS and LAS, if you explicitly declare this parameter then only this metric will be evaluated")
	parser.add_argument("--dependencies", "-dep", default=False, action="store_true",
	                     help="Compute statistics of dependencies length of a word, default=False")
	parser.add_argument("--root", "-root", default=False, action="store_true",
	                     help="Compute statistics of distance to the root, default=False")	                   
	parser.add_argument("--child", "-c", default=False, action="store_true", 
								help="Compute statistics of number of modifier siblings, default=False")                     
	parser.add_argument("--non_projectivity", "-p", default=False, action="store_true", 
								help="Compute statistics for degree of non projectivity, default=False")
	parser.add_argument("--part_of_speech", "-pos", default=False, action="store_true", 
								help="""Compute statistics for part of speech. LAS for the modifier word in a 
										dependency relation, default=False""")
	parser.add_argument("--deep_pos", "-dpos", type=str, default='ALL',
                        help="""Deepen the analysys on part of speech splitting the set, for example inside the pronoun category you have
                        		  multiple kind of pronoun like personal, relatice, clitic... value acceptable as parameter:
                        		  -VERB -NOUN -PRON -ADJC -ADVR -PREP -CONJ""")
	parser.add_argument("--dependency_relation", "-deprel", default=False, action="store_true", 
								help="""Compute statistics for dependency relation. LAS for the dependency relation arc. default=False""")
	parser.add_argument("--pos_occur", "-po", default=False, action="store_true",
								help=""" Type this parameter to see the tables of all the pos and relative occurences""")
	parser.add_argument("--deprel-occur", "-do", default=False, action="store_true",
								help=""" Type this parameter to see the table of all the deprel and relative occurences
											by default deprel with less than 25 occurences are excluded to avoid graph 
											explosion. You can change this with --min_occurences parameters. Set to 0 to have all""")
	parser.add_argument("--min_occurences", "-minocc", type=int, default=25)	
	args = parser.parse_args()

	# Load datasets
	gold_ud = load_conllu_file("gold.conllu")
	system_ud = load_conllu_file("syst.conllu")
	alignment = align_words(gold_ud.words, system_ud.words)
	
	pos_dep_occurences(gold_ud.words, args.deprel_occur, args.pos_occur, args.min_occurences) # change gold_ud.words to system_ud.words to explore other ds
	
	# Lists for graphs
	VALUES = []
	PRECISION = []
	RECALL = []	
	F1 = []
	# Metrics
	keys_fn = [("UAS_LEN", lambda w, ga: ga(w.parent)), ("LAS_LEN",lambda w, ga: (ga(w.parent), w.columns[DEPREL]))]
	if args.LAS: 
		keys_fn = [keys_fn[1]]
	if args.UAS: 
		keys_fn = [keys_fn[0]]
   
   # start the statistics for DEPENDENCY DISTANCE
	if args.dependencies: 
		# itera sulle metriche
		for key in keys_fn:	
		# inizializza liste per i grafici		
			ClearListe(VALUES, PRECISION, RECALL)
	
		# stampa statistiche		
			for d in range(1, args.maxdistance+1):
				evaluation = { key[0] : alignment_score(alignment, key_fn=key[1], filter_fn=lambda w: (dependency_distance(w)) == d)}
				evaluation_print(key[0], d)
	
				VALUES.append(d)
				PRECISION.append(evaluation[key[0]].precision)
				RECALL.append(evaluation[key[0]].recall)
			precision_recall_graph(VALUES, PRECISION, RECALL, "DEPENDENCY LENGTH", "DEPENDENCY EVALUATION", key[0])
				
	# start the statistics for DISTANCE TO THE ROOT					
	if args.root:
		# itera sulle metriche
		for key in keys_fn:
		# inizializza liste per i grafici
			ClearListe(VALUES, PRECISION, RECALL)
	
		# stampa statistiche
			for d in range(1, args.maxdistance+1):
				evaluation = { key[0] : alignment_score(alignment, key_fn = key[1], filter_fn=lambda w: udword_distance_to_root(w) == d)}   
				evaluation_print(key[0], d)
		
				VALUES.append(d)
				PRECISION.append(evaluation[key[0]].precision)
				RECALL.append(evaluation[key[0]].recall)
			precision_recall_graph(VALUES, PRECISION, RECALL, "DISTANCE TO ROOT", "DEPENDENCY EVALUATION", key[0])  
   
	# start the statistics for NUMBER OF MODIFIER SIBLINGS
	if args.child:
		# itera sulle metriche
		for key in keys_fn:
		# inizializza liste per i grafici
			ClearListe(VALUES, PRECISION, RECALL)
		
		# stampa statistiche
			for grado in range(1, args.maxgrade):
				evaluation = { key[0] : alignment_score(alignment, key_fn = key[1], filter_fn=lambda w: word_grade(w) == grado)}  
				evaluation_print(key[0], grado)
				
				VALUES.append(grado)
				PRECISION.append(evaluation[key[0]].precision)
				RECALL.append(evaluation[key[0]].recall)
			precision_recall_graph(VALUES, PRECISION, RECALL, "NUMBER OF MODIFIER SIBLINGS", "DEPENDENCY PRECISION", key[0])
	
	# start the statistics for NON-PROJECTIVE ARC DEGREE
	if args.non_projectivity:
		# inietta ai dataset le proprietà per computare la metrica
		non_projective_grade(gold_ud)
		non_projective_grade(system_ud)
		# itera sulle metriche
		for key in keys_fn:
		# inizializza liste per i grafici
			ClearListe(VALUES, PRECISION, RECALL)
		
		# stampa statistiche
			for grado in range(0, args.maxgrade):
				evaluation = { key[0] : alignment_score(alignment, key_fn = key[1], filter_fn=lambda w: w.non_projectivity == grado)}  
				evaluation_print(key[0], grado)
				
				VALUES.append(grado)
				PRECISION.append(evaluation[key[0]].precision)
				RECALL.append(evaluation[key[0]].recall)
			precision_recall_graph(VALUES, PRECISION, RECALL, "NON-PROJECTIVE ARC DEGREE", "DEPENDENCY PRECISION", key[0])
				
	if args.part_of_speech:
		
		# verb, noun, pronoun, adj, adv, adp, conj 
		
		tagset = {
					 'VERB':('VA','V','VM'),'NOUN':('SP','S'),'PRON':('DD','PE','PR','PC','PI','PP','PD','PQ'),
					 'ADJC':('A', 'AP'),'ADVR':('B','BN'),'PREP':('E'),'CONJ':('CC','CS'), 'DETR':(), 'PNCT':('FC','FB','FS','FF'),
					 'DETR':('T','RI','DR','DE','RD','DQ')
					}
	 
		ClearListe(VALUES, F1)	 
		#set metric to LAS
		key = ("LAS",lambda w, ga: (ga(w.parent), w.columns[DEPREL]))
		#itera sulle part of speech			
		filtro =  lambda w: ((w.columns[4] if w.parent is not None else 'NONE') in tagset[pos])
		tags = tagset.keys()
		
		#change condition if deep inspection is invoked
		if args.deep_pos != 'ALL': 
			filtro = lambda w: ((w.columns[4] if w.parent is not None else 'NONE') == pos)
			tags = tagset[args.deep_pos]	
		
		for pos in tags:
			evaluation = { key[0] : alignment_score(alignment, key_fn = key[1], filter_fn=filtro)}	
			pos_deprel_evaluation_print(key[0], pos)
			
			VALUES.append(pos)
			F1.append(evaluation[key[0]].f1)
		pos_deprel_bar_graph(VALUES, F1, 'LAS (F1 SCORE)', 'POS tags')

	if args.dependency_relation:
		
		depreltagset = [
					 {'det', 'punct', 'ccomp', 'appos', 'cop', 'advcl', 'flat', 'vocative'}, {'amod', 'iobj', 'xcomp',
					 'nsubj', 'advmod', 'expl', 'aux'}, {'case', 'obl', 'parataxis', 'cc', 'root', 'nmod', 'mark'},
					 {'nummod', 'acl', 'csubj', 'obj', 'compound', 'conj', 'fixed'}
					]
			 		
		key = ("LAS",lambda w, ga: (ga(w.parent), w.columns[DEPREL]))
		for batch in depreltagset:
			ClearListe(VALUES, PRECISION, RECALL)			
			for depreltag in batch:
				evaluation = { key[0] : alignment_score(alignment, key_fn = key[1], filter_fn=lambda w: w.columns[7] == depreltag)}
				pos_deprel_evaluation_print(key[0], depreltag)
			
				VALUES.append(depreltag)
				PRECISION.append(evaluation[key[0]].precision)
				RECALL.append(evaluation[key[0]].recall)			
			pos_deprel_bar_graph(VALUES, PRECISION, 'LAS (PRECISION)', 'DEPREL tags')		
			pos_deprel_bar_graph(VALUES, PRECISION, 'LAS (RECALL)', 'DEPREL tags')				