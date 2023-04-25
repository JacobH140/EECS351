import nltk
from lark import Lark, Visitor
from itertools import product


def create_syntax_tree(sentence):
    """generates a traversable x-bar tree out of input.
    if input is not a valid sentence ('tense phrase'), will attempt to
    parse it as whatever 'phrase' it is (e.g., 'verb phrase').

    The actual generated tree will not account for: punctuation,
    symbols, interjections. prior to parsing, these will be removed.
    the final sonification WILL account for them, however, in specific
    ways outlined in [[createXBarTree]].md

    If that fails (as it often will when first testing the program since
    toy input such as 'cat bat' will be given) will return, instead
    of a data structure representing the tree, an int which is the number
    of words given. the function calling this one can then decide how to handle
    these cases."""

    tree = find_valid_parse(sentence)
    print(tree.pretty())
    return tree


def find_valid_parse(sentence):
    """In an ideal world, every input sentence would be a perfect CP with no 'X's.
    Sometimes, though, a VP will be given instead, or there will be some particle whose POS
    we cannot determine the first time around (likely a P). This function exists for those cases,
    more or less trying different starting points and different POS-values for different X's to see what works
    (usually there will only be a couple options)."""

    """This function has 3 stages. If it finds a successful tree in a given stage, it will not proceed to the next.
    Generally, the first stage yields the quickest and most accurate tree, but is also more likely to fail. Conversely, 
    the third stage is less likely to fail but may take a very long time to generate suboptimal trees."""

    prepared_sentence, removed_material = prepare_sentence(sentence)

    start_options = ['cp', 'tp', 'dp', 'np', 'vp', 'pp', 'adjp', 'advp']  # ordered most-to-least likely given no other info
    pos_options = ['P', 'ADV', 'D', 'N', 'V', 'ADJ', 'C', 'T']  # ordered most-to-least likely (note: 'not' is an ADV)

    output = -1

    if not any([x[1] == 'X' for x in prepared_sentence]):  # i.e., if there are no Xs to worry about
        output = try_except_lark_parse_helper(prepared_sentence, start_options, sentence)
    else:  # in this case, there ARE Xs to worry about...
        output = unknown_pos_try_except_lark_parse_helper(prepared_sentence, start_options, sentence, pos_options)

    # if output == -1:  # now if we STILL have not found a tree, try more... ambitious combinations
    #    # the (rather poor) idea: find ALL possible POS tagging combos that yield valid trees, then
    #    # choose one most similar to our original POS tagging. time complexity = terrible; number
    #    # of times called = terribly low... hopefully these offset each other
    #    print("ENTERED STAGE 3")
    #    edited_sentences = []
    #    successful_combinations = []  # coindexed with edited sentences
    #    smaller_pos_options = ['ADJ', 'ADV']  # for speeding up debugging mostly
    #    combinations = product(smaller_pos_options, repeat=len(prepared_sentence))
    #    for combination in combinations:
    #        prepared_sentence_very_edited = [tuple([prepared_sentence[i][0], pos]) for i, pos in enumerate(combination)]
    #      try:
    #          edited_sentences.append(try_except_lark_parse_helper(prepared_sentence_very_edited, start_options, sentence))
    #          successful_combinations.append(combination)
    #      except Exception:
    #          print("State 3 of parsing, failed to parse ")
    #            print(prepared_sentence_very_edited)
    #
    #    # now find an edited sentence POS tagging which closely matches the original:
    #    if not edited_sentences:
    #        print("Apparently, no valid POS combinations were found which yield a valid parsing...?")
    #    else:
    #        sentence_pos_tags = [tag for word, tag in prepared_sentence]
    #        prepared_sentence_pos_set = set(sentence_pos_tags)
    #        best_combo_so_far = successful_combinations[0]
    #        best_edited_sentence_so_far = edited_sentences[0]
    #        for index, edited_sentence in enumerate(edited_sentences):
    #            if len(set(successful_combinations[index]).intersection(prepared_sentence_pos_set)) \
    #                    > len(set(best_combo_so_far).intersection(prepared_sentence_pos_set)):
    #                best_combo_so_far = successful_combinations[index]
    #                best_edited_sentence_so_far = edited_sentence
    #
    #        output = best_edited_sentence_so_far

    return output


def try_except_lark_parse_helper(prepared_sentence, start_options, sentence):
    output_parse = -1
    for start_option in start_options:
        try:
            output_parse = grammar(prepared_sentence, start_option, sentence)
            break
        except Exception:
            print("Failed to parse as " + start_option + ". Will try another phrase structure.")
            if start_option == 'advp':
                print("FAILED TO PARSE TREE. What to do in this situation is yet to be implemented.")
        try:
            output_parse = grammar(sentence.lower(), start_option, sentence)
            break
        except Exception:
            print("also tried all lower case, and failed...")
            continue
    return output_parse

def unknown_pos_try_except_lark_parse_helper(prepared_sentence, start_options, sentence, pos_options):
    output = -1
    Xs = [i for i in prepared_sentence if i[1] == 'X']  # grab the 'X' entries
    combinations = product(pos_options, repeat=len(Xs))
    for combination in combinations:
        x_index = 0
        prepared_sentence_edited = []
        for i in prepared_sentence:
            if i[1] == 'X':
                prepared_sentence_edited.append(tuple([i[0], combination[x_index]]))
                x_index += 1
            else:
                prepared_sentence_edited.append(i)
        for start_option in start_options:
            try:
                output = grammar(prepared_sentence_edited, start_option, sentence)
                return output
            except Exception:
                print("Failed to parse: ")
                print(prepared_sentence_edited)
                print("Will try another set of values for the items with unknown POS")
                continue


def grammar(prepared_sentence, start, sentence):
    print(prepared_sentence)
    # for debugging:
    if not isinstance(prepared_sentence, list):
        x, y = prepare_sentence(prepared_sentence)
        prepared_sentence = x
    assert isinstance(prepared_sentence, list)
    terminals = create_terminals_for_sentence(prepared_sentence)
    lark_input = ''' %import common.WORD
                %import common.WS
                WORDTOK: WORD WS? \n''' + terminals + '''
                cp: c_bar
                c_bar: C? tp
                tp: dp t_bar 
                t_bar: T? vp
                vp: ADV? v_bar
                v_bar: v_bar pp | v_bar advp | advp v_bar | V dp? | v_bar adjp | vp
                dp: dp? d_bar
                d_bar: D np?
                np: n_bar
                n_bar: adjp n_bar | n_bar pp | N pp?
                pp: p_bar
                p_bar: p_bar pp | advp p_bar | P dp? | P np?
                advp: adv_bar 
                adv_bar: advp adv_bar | ADV pp?
                adjp: adj_bar | adjp CONJ adjp
                adj_bar: adjp adj_bar | ADJ pp?
                
                xp: xp CONJ xp
                x: x CONJ x '''
    # note: removed the tp cp t_bar option as it just... breaks the parsing. not ideal, but a worthy tradeoff, i think
    # note: added the vp v_bar option, though i have yet to confirm with anyone whether doing so is correct or optimal
    print(lark_input)
    lexicon_friendly_sentence = " ".join([word for word, tag in prepared_sentence])  # e.g., we need to give the parser "He walks" or "he walks", not "He walks!" "
    return Lark(lark_input, start=start).parse(lexicon_friendly_sentence)  # if an exception is thrown, it will be here


def create_terminals_for_sentence(prepared_sentence):
    dictionary = {'N': [], 'D': [], 'V': [], 'ADJ': [], 'ADV': [], 'P': [], 'T': [], 'C': [], 'CONJ': [], 'X': []}

    output = ""

    for word, pos in prepared_sentence:
        dictionary[pos].append(word)

    for pos_key in dictionary:
        instances = "("
        if dictionary[pos_key]:
            instances += '"' + dictionary[pos_key][0] + '"'
        else:
            instances = '"_"'

        for i in range(len(dictionary[pos_key]) - 1):
            instances += " | " + '"' + dictionary[pos_key][i + 1] + '"'

        if instances != '"_"':
            instances += ")"
        instances += " WS?" + '\n'

        output += pos_key + ": " + instances
    return output


def prepare_sentence(sentence, lower=False):
    # first, tokenize
    tokens = nltk.word_tokenize(sentence)
    words_and_tags = assign_pos_tags(tokens)
    removed_material = []  # contains tuples describing (where, who) is getting removed
    output = []  # where those NOT being removed go. (word, tag) tuples with bad ones removed.
    accepted_tags = ['ADJ', 'ADV', 'CONJ', 'DET', 'NOUN', 'PROPN', 'PRON', 'ADP'
                     , 'AUX', 'VERB', 'SCONJ']
    grammar_pos_map = {'ADJ': 'ADJ',
                       'ADV': 'ADV',
                       'CONJ': 'CONJ',
                       'DET': 'D',
                       'NOUN': 'N',
                       'PROPN': 'N',
                       'PRON': 'D',
                       'ADP': 'P',
                       'AUX': 'V',
                       'VERB': 'V',
                       'SCONJ': 'C',
                       'X': 'X',
                       'T': 'T'
                       }

    # find out who we should remove, but remember where they were
    for index, tup in enumerate(words_and_tags):
        word = tup[0]
        tag = tup[1]

        # handle the noun-noun case, where it is safe to treat the preceding noun as an ADJ
        if words_and_tags[index][1] in grammar_pos_map.keys() and words_and_tags[index] != words_and_tags[-1] and \
                words_and_tags[index + 1][1] in grammar_pos_map.keys() and \
                grammar_pos_map[words_and_tags[index + 1][1]] == 'N' and grammar_pos_map[tag] == 'N':
            output.append((tuple([word, 'ADJ'])))

        elif words_and_tags[index][1] in grammar_pos_map.keys() and words_and_tags[index] != words_and_tags[-1] and \
                words_and_tags[index + 1][1] in grammar_pos_map.keys() and \
                grammar_pos_map[words_and_tags[index + 1][1]] == 'V' and grammar_pos_map[tag] == 'V':
            output.append((tuple([word, 'T'])))  # this will sometimes lead to a technically incorrect POS tagging,
            # but nevertheless correct syntax tree

        elif word.lower() == "'s":  # handle the 's-genitive as DET where necessary, as DP-hypothesis suggests
            output.append(tuple([word, 'DET']))

        # TODO: explicitly handle complementizers? there's a chance the grammar will still hate them though...

        elif tag not in accepted_tags:
            instructions = handle_unaccepted_tag(index, tag, words_and_tags, grammar_pos_map)
            if instructions == "remove":
                removed_material.append(tuple([index, word, tag]))
            else:  # else, handle_unaccepted_tag plans to replace the tag with an accepted one
                # in this case, instructions will return the replacement tag
                output.append(tuple([word, instructions]))
        else:
            output.append(tuple([word, tag]))

    # finally, map each pos we've found to their equivalent in the grammar
    output = [tuple([word_tag[0], grammar_pos_map[word_tag[1]]]) for word_tag in output]
    return output, removed_material


def handle_unaccepted_tag(index, tag, words_and_tags, grammar_pos_map):
    # some casework for handling different situations
    """match tag:
        case 'NUM':  # numbers are generally adjs or dets when followed by nouns or adj or adv, nouns otherwise
            if words_and_tags[index] != words_and_tags[-1]:  # so we don't go off the edge
                if grammar_pos_map[words_and_tags[index+1][1]] == 'N' or \
                        grammar_pos_map[words_and_tags[index+1][1]] == 'ADJ':
                    return 'ADJ'
                else:
                    return 'NOUN'
            else:
                return 'NOUN'
        case 'INTJ':
            return "remove"
        case 'SYM':
            return "remove"
        case 'PRT':
            return 'X'  # we want it to be treated how an unknown is treated
        case 'X':
            return 'X'  # just to be comprehensive

    # if we have gotten here, it means we got an otherwise weird tag. Remove it: """

    if tag == 'NUM':
        if words_and_tags[index] != words_and_tags[-1]:  # so we don't go off the edge
            if grammar_pos_map[words_and_tags[index + 1][1]] == 'N' or \
                    grammar_pos_map[words_and_tags[index + 1][1]] == 'ADJ':
                return 'ADJ'
            else:
                return 'NOUN'
        else:
            return 'NOUN'
    elif tag == 'INTJ':
        return "remove"
    elif tag == 'SYM':
            return "remove"
    elif tag ==  'PRT':
        return 'X'  # we want it to be treated how an unknown is treated
    elif tag == 'X':
        return 'X'  # just to be comprehensive

    return "remove"


def assign_pos_tags(word_tokens):
    """input: list of words in sentence (word tokens)
       output: list of tuples of the form(word, POS)"""
    tags = nltk.pos_tag(word_tokens, tagset='universal')
    return tags


if __name__ == "__main__":
    # sent = "John's big idea isn't all that bad."
    sent = "there are two apples, and that one should be my mother's!"
    # assign_pos_tags(nltk.word_tokenize(sent))
    # print(prepare_sentence(sent))
    # print(grammar("Walked a mile", 'vp', "Walked a mile"))
    # print((find_valid_parse("She walks").pretty()))
    # gen = (find_valid_parse("Two red pandas").iter_subtrees_topdown())
    # for i in gen:
    #    print(i) # there's gotta be some nice way to traverse similar to this
    print(find_valid_parse("The book on the shelf").pretty())
