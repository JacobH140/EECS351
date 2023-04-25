from music21 import key, roman, stream
import lark
import syntax_tree_v2
import sonify_word


def sonify_sentence(sentence, mode, audio_on=True):
    """TODO: improve this docstring
    traverses the tree to assign chords to words based on x-bar theory then adds back in the material that was removed during the
    tree-generation process, deciding what chords go with what additional material etc"""
    output = stream.Stream()

    tree = syntax_tree_v2.create_syntax_tree(sentence)
    prepared_sentence, additional_material = syntax_tree_v2.prepare_sentence(sentence)  # for adding back in INTJs, punctuation, etc
    # print(tree.pretty())
    print(prepared_sentence)
    print(additional_material)
    # print(get_head_token(tree))
    # print(has_specifier(tree))
    # print(has_adjunct(tree))
    # print(has_complement(tree))
    # print(get_specifier(tree).pretty())
    # print(get_adjunct(tree).pretty())
    # print(get_complement(tree).pretty())
    chordal_parse = []
    traverse(tree, roman.RomanNumeral('I    '), mode, chordal_parse, len(prepared_sentence)) # 230328 messing around, started as I... music21 treats I in g minor as G for example, don't know if we rly want that
    print(chordal_parse)
    handle_additional_material(additional_material, chordal_parse)
    print("chordal parse:", chordal_parse)
    for h, m, w in chordal_parse:
        print("for loop over chordal parse:", h, m, w)
        output.append(sonify_word.sonify_word(h, m, w.strip(), audio_on))

    print("chordal parse:", chordal_parse)
    return output



def handle_additional_material(additional_material, chordal_parse):
    # CURRENTLY ASSUMES: additional material has nothing corresponding to the beginning of chordal_parse
    # (since, for example, starting a sentence with ! in english would be a strange (¡though not in spanish!)
    for w in additional_material:
        if w[2] == '.':  # punctuation: just keep the preceding chord
            i = w[0]
            chordal_parse.insert(i, tuple([chordal_parse[i-1][0], chordal_parse[i-1][1], w[1]]))
        else:  # modal mixture to indicate a departure from key (since we have a syntactically strange entity)
            i = w[0]
            chordal_parse.insert(i, tuple([chordal_parse[0][0], chordal_parse[0][1].minor, w[1]]))



def traverse(phrase, harmonic_function, mode, output, len_prepared_sentence):
    """main recursive function"""
    if not phrase:
        return None

    # BASE CASE: Traverse has been called on a head/leaf/single token
    if is_head(phrase):
        output.append(tuple([harmonic_function, mode, phrase]))  # to be passed into sonify_word()
        return None

    if len(output) == len_prepared_sentence:
        return None


    # RECURSIVE CASE — see diagram

    # first, check for specifier
    if has_specifier(phrase):
        specifier = get_specifier(phrase)
        h, m = specifier_transform(harmonic_function, mode)
        traverse(specifier, h, m, output, len_prepared_sentence)

    # next, check for adjunct. if it exists, make note of which side of head it's on
    has_ad = has_adjunct(phrase)
    if has_ad and adjunct_on_right(phrase):
        has_ad = "RIGHT"
    elif has_ad:
        has_ad = "LEFT"
    # note that has_ad is enum-flavored: can be False, "RIGHT", or "LEFT"

    has_comp = has_complement(phrase)  # also grab this

    if has_ad == "LEFT":
        adjunct_traverse_helper(phrase, harmonic_function, mode, output, len_prepared_sentence)
        traverse(get_head_token(phrase), harmonic_function, mode, output, len_prepared_sentence)  # call on head
        if has_comp:
            complement_traverse_helper(phrase, harmonic_function, mode, output, len_prepared_sentence)

    elif has_ad == "RIGHT":
        traverse(get_head_token(phrase), harmonic_function, mode, output, len_prepared_sentence)  # call on head
        if has_comp:
            complement_traverse_helper(phrase, harmonic_function, mode, output, len_prepared_sentence)
        adjunct_traverse_helper(phrase, harmonic_function, mode, output, len_prepared_sentence)

    else:  # else, has_ad must be False
        if has_comp:
            traverse(get_head_token(phrase), harmonic_function, mode, output, len_prepared_sentence)  # call on head
            complement_traverse_helper(phrase, harmonic_function, mode, output, len_prepared_sentence)
        else:
            traverse(get_head_token(phrase), harmonic_function, mode, output, len_prepared_sentence)  # call on head
    # traverse(prepared_sentence, get_head_token(phrase), harmonic_function, mode, output)


def complement_traverse_helper(phrase, harmonic_function, mode, output, len_prepared_sentence):
    complement = get_complement(phrase)
    h, m = complement_transform(harmonic_function, mode)
    traverse(complement, h, m, output, len_prepared_sentence)


def adjunct_traverse_helper(phrase, harmonic_function, mode, output, len_prepared_sentence):
    # solely for reducing code duplication in traverse()
    adjunct = get_adjunct(phrase)
    h, m = adjunct_transform(harmonic_function, mode)
    traverse(adjunct, h, m, output, len_prepared_sentence)


def specifier_transform(harmonic_function, mode):
    """TODO: EXPERIMENTATION-FRIENDLY
    determines what the harmonic_function and mode inputs of traverse should be when
    the phrase input is the specifier of another phrase"""
    new_function = roman.RomanNumeral((harmonic_function.scaleDegree + 4) % 7)  # up P5
    new_mode = mode
    return new_function, new_mode


def complement_transform(harmonic_function, mode):
    """TODO: EXPERIMENTATION-FRIENDLY
    determines what the harmonic_function and mode inputs of traverse should be when
    the phrase input is the complement of another phrase"""

    # also consider relative minor... the idea for now is just to get the generation
    # working as expected, these parameters can be adjusted wrt discussion w/ others
    # later

    new_function = roman.RomanNumeral((harmonic_function.scaleDegree + 4) % 7)
    new_mode = mode
    return new_function, new_mode


def adjunct_transform(harmonic_function, mode):
    """TODO: EXPERIMENTATION-FRIENDLY
    determines what the harmonic_function and mode inputs of traverse should be when
    the phrase input is the complement of another phrase"""
    new_function = roman.RomanNumeral('I')
    new_mode = mode.transpose('P5')
    return new_function, new_mode


def get_specifier(phrase):
    # INVARIANT: must have a specifier, (see has_specifier())
    return phrase.children[0]


def get_adjunct(phrase):
    # INVARIANT: must have an adjunct, (see has_adjunct())
    # adjuncts are weird, since they can be either left or right child of the bar,
    # this solution attempts to account for that
    top_bar_level = phrase.children[-1]
    if adjunct_on_right(phrase):
        return top_bar_level.children[-1]
    else:
        return top_bar_level.children[0]


def adjunct_on_right(phrase):
    # INVARIANT: must have an adjunct, (see has_adjunct())
    top_bar_level = phrase.children[-1]
    if isinstance(top_bar_level.children[0].children[0], lark.Token):
        return True
    return False


def get_complement(phrase):
    # INVARIANT: must have a complement, (see has_complement())
    has_ad = has_adjunct(phrase)
    if not has_ad:
        return phrase.children[-1].children[-1]
    else:
        return phrase.children[-1].children[0].children[-1]


def get_head_token(phrase):
    """to be honest, this one is a little trippy — good to review when debugging
    i still think it does not work 230226 Edit: I think that it did end up working eventually, thought
    can't be certain..."""


    # note that tree.data gives a string w/ the root of the tree, if that is what i want at all

    # note that cp and tp are allowed to have 'null' heads for our purposes

    right = phrase.children[-1]  # rightmost child
    if right.children and isinstance(right.children[0], lark.Token):
        return right.children[0]
    else:
        # a bit janky, but it shouldn't fail given the grammar structure:
        # to account for the case where adjunct phrase is on LHS of head (e.g. "The big book"),
        # check instances in one direction, and upon failure try the other direction
        if isinstance(right.children[0].children[0], lark.Token):
            return right.children[0].children[0]
        else:
            print(phrase)
            try:
                assert isinstance(right.children[-1].children[0], lark.Token)
            except AssertionError:
                if phrase.data == 'tp' or phrase.data == 'cp':
                    print("tp/cp")
                    return None
                else:
                    print(phrase.data)
                    assert False
            return right.children[-1].children[0]


def has_specifier(phrase):
    # MUST be called on a PHRASE. If accidentally called on a bar or a head, things will break
    # print(phrase)

    if len(phrase.children) == 2:
        return True
    return False


def has_adjunct(phrase):
    # "sister of a bar, daughter of a bar"
    right_tree = phrase.children[-1]
    if not any([isinstance(child, lark.Token) for child in right_tree.children]):
        return True
    return False


def has_complement(phrase):
    has_ad = has_adjunct(phrase)
    if not has_ad:
        right_tree = phrase.children[-1]
        return len(right_tree.children) == 2
    else:
        second_bar_tree = phrase.children[-1].children[0]
        return len(second_bar_tree.children) == 2


def is_head(phrase):
    # helper function for tree traversal
    if isinstance(phrase, lark.Token):
        return True
    return False


if __name__ == "__main__":
    sent = "hello world!"
    output = sonify_sentence(sent, key.Key('c'))
    print(output)
    output.write("midi", sent + ".mid")
