import numpy as np


def find_mask_words(abstracts, word, verbose=True):
    """Creates a mask for abstracts containing a certain word.
    Creates several masks of the size of `abstracts` for instances containing the words in `words`. Also it prints how many instances contain each word, in its capitalized, uncapitalized versions, and total.
    If I query the word "retina" it will also give back words like "retinal" because it has to match the beginning of the word but it can still be a substring of another full word.
    
    Parameters
    ----------
    abstracts : pandas dataframe of str
        All texts (in this case abstracts).
    words : str
        str of the word/phrase to be queried.
    verbose : bool, optional
        If True, prints the number of times the word appears in its different forms in the abstracts collection.

    Returns
    -------
    mask : array-like of bool
        Mask.

    """

    sub1 = " " + word
    sub2 = word.capitalize()

    indexes1 = abstracts.str.find(sub1)
    indexes2 = abstracts.str.find(sub2)

    mask = (indexes1 != -1) | (indexes2 != -1)

    if verbose == True:
        print(
            f"Number of papers with uncapitalized word '{word}': ",
            len(np.where(indexes1 != -1)[0]),
        )
        print(
            f"Number of papers with capitalized word '{word}': ",
            len(np.where(indexes2 != -1)[0]),
        )
        print(f"Number of total papers with word '{word}': ", len(np.where(mask)[0]))

    return mask
