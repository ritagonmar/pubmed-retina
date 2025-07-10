import glasbey
import numpy as np
import re
import pandas as pd
from typing import List, Dict, Union
import seaborn as sns


def improved_coloring(journals, dict_words_colors):
    """Creates coloring based on words appearing in a list of documents.
    It creates an array with colors, assigning a color to each paper depending on whether it contains a word in its journal title from the keys in ` dict_words_colors`.

    IMPORTANT REMARK: if the journal name contains two words belonging to the word list, the color of the word
    located the latest in the list will be assigned to it (first, the first word's color is assigned and then
    the second overwrites the first).

    Parameters
    ----------
    journals : dataframe of str
        Dataframe with the journal names of the papers, or any other corpus where to look for the words.
    dict_words_colors : dict
        Dictionary matching words to colors (legend). The keys are the words and the values are the colors.


    Returns
    -------
    labels_with_unlabeled : list of str fo len (n_journals)
        List or labels (words) for all instances including label 'unlabeled'.
    colors : array
        Colors for each paper.

    See Also
    --------
    automatic_coloring

    """

    words = dict_words_colors.keys()
    labels = np.empty(len(journals))

    for i, wrd in enumerate(words):

        word_may = wrd.capitalize()
        word_min = " " + wrd

        indexes1 = journals.str.find(word_may)
        indexes2 = journals.str.find(word_min)

        labels = np.where((indexes1 != -1) | (indexes2 != -1), wrd, labels)

    # create colors
    colors = np.vectorize(dict_words_colors.get)(labels)

    # add grey to the rest of papers
    colors = np.where(colors == None, "lightgrey", colors)
    colors = np.where(colors == "None", "lightgrey", colors)

    # change 0 for 'unlabeled'
    labels_with_unlabeled = np.where(colors == "lightgrey", "unlabeled", labels)

    return labels_with_unlabeled, colors


# More efficient version for very large datasets using vectorized operations
def label_texts_with_animals_vectorized(
    texts: pd.Series,
    animals_to_check: List[str],
    species_forms: Dict[str, List[str]],
    multilabel: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Vectorized version for labeling texts based on animal mentions.
    This version is more efficient for very large datasets.

    Args:
        texts: Pandas Series containing the texts to analyze
        animals_to_check: List of animal keys to search for
        species_forms: Dictionary mapping each animal key to its various forms
        multilabel: If True, returns a DataFrame with animal presence indicated by True/False
                   If False, returns a Series with the first animal found for each text

    Returns:
        If multilabel=False: Pandas Series with single label for each text
        If multilabel=True: Pandas DataFrame with boolean columns for each animal
    """

    # Filter to only relevant animals
    filtered_species_forms = {
        k: v for k, v in species_forms.items() if k in animals_to_check
    }

    # Ensure texts are strings
    texts = texts.fillna("").astype(str)

    if multilabel:
        # Create a DataFrame to store results, initialized with all False
        results = pd.DataFrame(False, index=texts.index, columns=animals_to_check)

        # Process each animal
        for animal, forms in filtered_species_forms.items():
            # Create regex patterns for each form
            patterns = [r"\b" + re.escape(form) + r"\b" for form in forms]
            combined_pattern = "|".join(patterns)

            # Set True for texts that contain this animal
            results[animal] = texts.str.contains(
                combined_pattern, case=False, regex=True
            )

    else:
        # Initialize results with "unlabeled"
        results = pd.Series(["unlabeled"] * len(texts), index=texts.index)

        # Process each animal in priority order
        for animal, forms in filtered_species_forms.items():
            # Create regex patterns for each form
            patterns = [r"\b" + re.escape(form) + r"\b" for form in forms]
            combined_pattern = "|".join(patterns)

            # Create a mask for texts that have this animal and haven't been labeled yet
            mask = (texts.str.contains(combined_pattern, case=False, regex=True)) & (
                results == "unlabeled"
            )

            # Apply the animal label where the mask is True
            results[mask] = animal

    return results


def create_animal_lookup(taxon_animals):
    """
    Creates a reverse lookup dictionary: animal -> taxonomic group.
    This is done once for efficiency with large lists.
    """
    lookup = {}
    for taxon_group, animals in taxon_animals.items():
        for animal in animals:
            lookup[animal] = taxon_group
    return lookup


def color_mapping(labels, legend_dict):
    colors_class = np.vectorize(legend_dict.get)(labels)

    # add grey to the rest of papers
    colors_class = np.where(colors_class == None, "lightgrey", colors_class)
    colors_class = np.where(colors_class == "None", "lightgrey", colors_class)
    return colors_class


def create_color_mapping(labels, plot_palette=False, palette_kwargs=None):
    if palette_kwargs is None:
        palette_kwargs = {
            "lightness_bounds": (20, 75),
            "chroma_bounds": (50, 90),
        }

    palette = glasbey.create_palette(
        palette_size=len(np.unique(labels)), **palette_kwargs
    )
    if plot_palette:
        sns.palplot(palette)

    legend_dict = dict(zip(np.unique(labels), palette))
    legend_dict["unlabeled"] = "lightgrey"

    colors_animals = color_mapping(labels, legend_dict)
    return colors_animals, legend_dict
