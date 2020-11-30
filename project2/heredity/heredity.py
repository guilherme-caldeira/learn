import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Initialize probabilities to be calculated
    prob_joint = prob_gene_person = prob_trait_person = 1.0
    prob_mother = prob_not_mother = prob_father = prob_not_father = 1.0

    # Iterate through each person in the dictionary
    for person, data in people.items():
        mother = data["mother"]
        father = data["father"]

        # Get the probability of person having a particular number of the gene
        person_gene = get_gene(person, one_gene, two_genes)

        # If it is possible to determine the person's mother and father,
        # then, it is necessary to calculate the probability
        if mother is not None and father is not None:
            mother_gene = get_gene(mother, one_gene, two_genes)
            father_gene = get_gene(father, one_gene, two_genes)
            prob_mother = get_prob_parent(mother_gene)
            prob_father = get_prob_parent(father_gene)
            prob_not_mother = get_prob_not_parent(mother_gene)
            prob_not_father = get_prob_not_parent(father_gene)
            if person_gene == 2:
                prob_gene_person = prob_mother * prob_father
            elif person_gene == 1:
                prob_gene_person = (prob_mother * prob_not_father) + (prob_not_mother * prob_father)
            elif person_gene == 0:
                prob_gene_person = prob_not_mother * prob_not_father
        # Else, the value is picked up from PROBS
        else:
            dict_gene = {
                o_k: {i_k: i_v for i_k, i_v in o_v.items() if i_k == person_gene}
                    for o_k, o_v in PROBS.items() if o_k == "gene"}
            prob_gene_person = dict_gene["gene"][person_gene]
        
        # Get the probability of the person does or does not having a particular trait
        # This value is picked up from PROBS
        person_trait = get_trait(person, have_trait)
        dict_trait = {
            o_k: {m_k: {i_k: i_v for i_k, i_v in m_v.items() if i_k == person_trait}
                    for m_k, m_v in o_v.items() if m_k == person_gene}
                for o_k, o_v in PROBS.items() if o_k == "trait"}
        prob_trait_person = dict_trait["trait"][person_gene][person_trait]
        
        prob_joint = prob_gene_person * prob_trait_person * prob_joint
    
    return prob_joint


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        person_gene = get_gene(person, one_gene, two_genes)
        person_trait = get_trait(person, have_trait)
        old_p = probabilities[person]["gene"][person_gene]
        probabilities[person]["gene"][person_gene] = p + old_p
        old_p = probabilities[person]["trait"][person_trait]
        probabilities[person]["trait"][person_trait] = p + old_p

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    dict_gene = {}
    dict_trait = {}
    for person in probabilities:
        # Loop through all possible values for "gene" to normalize them
        for i in range(len(probabilities[person]["gene"])):
            dict_gene[i] = probabilities[person]["gene"][i]
        for i in range(len(probabilities[person]["gene"])):
            probabilities[person]["gene"][i] = dict_gene[i] / sum(dict_gene.values())
        # Loop through all possible values for "trait" to normalize them
        for i in range(len(probabilities[person]["trait"])):
            dict_trait[i] = probabilities[person]["trait"][i]
        for i in range(len(probabilities[person]["trait"])):
            probabilities[person]["trait"][i] = dict_trait[i] / sum(dict_trait.values())


def get_gene(person, one_gene, two_genes):
    if person in one_gene:
        return 1
    if person in two_genes:
        return 2
    else:
        return 0


def get_prob_parent(number_of_genes):
    if number_of_genes == 1:
        return 0.5
    elif number_of_genes == 2:
        return 1 - PROBS["mutation"]
    else:
        return PROBS["mutation"]


def get_prob_not_parent(number_of_genes):
    if number_of_genes == 1:
        return 0.5
    elif number_of_genes == 2:
        return PROBS["mutation"]
    else:
        return 1 - PROBS["mutation"]


def get_trait(person, have_trait):
    if person in have_trait:
        return True
    else:
        return False


if __name__ == "__main__":
    main()
