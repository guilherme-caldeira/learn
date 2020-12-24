import csv
import sys
from numpy.core.numeric import Infinity, NaN

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Define the evidence list
    evidence = []

    # Define the label list
    label = []

    # Load shopping.csv
    with open(filename, encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Add values to the evidence list
            evidenceRowList = create_list(row)
            evidence.append(evidenceRowList)

            # Add values to the label list
            lab = true_or_false(row["Revenue"])
            label.append(lab)
    
    return (evidence, label)


def create_list(row):
    """
    Given a row from the csv file, return a list to be added to evidence.
    It converts all values from the csv file to integer or float.
    """
    evidence_list = []
    adm = int(row["Administrative"])
    admDuration = float(row["Administrative_Duration"])
    inf = int(row["Informational"])
    infDuration = float(row["Informational_Duration"])
    prdRel = int(row["ProductRelated"])
    prdRelDuration = float(row["ProductRelated_Duration"])
    bounce = float(row["BounceRates"])
    exit = float(row["ExitRates"])
    page = float(row["PageValues"])
    special = float(row["SpecialDay"])
    month = month_to_int(row["Month"])
    oper = int(row["OperatingSystems"])
    browser = int(row["Browser"])
    region = int(row["Region"])
    traffic = int(row["TrafficType"])
    visitor = visit_to_int(row["VisitorType"])
    weekend = true_or_false(row["Weekend"])
    evidence_list.append(adm)
    evidence_list.append(admDuration)
    evidence_list.append(inf)
    evidence_list.append(infDuration)
    evidence_list.append(prdRel)
    evidence_list.append(prdRelDuration)
    evidence_list.append(bounce)
    evidence_list.append(exit)
    evidence_list.append(page)
    evidence_list.append(special)
    evidence_list.append(month)
    evidence_list.append(oper)
    evidence_list.append(browser)
    evidence_list.append(region)
    evidence_list.append(traffic)
    evidence_list.append(visitor)
    evidence_list.append(weekend)

    return evidence_list


def visit_to_int(visitor_type):
    switcher = {
        "Returning_Visitor":1,
        "New_Visitor":0,
        "Other":0
    }

    return switcher.get(visitor_type)


def true_or_false(input):
    switcher = {
        "TRUE":1,
        "FALSE":0
    }

    return switcher.get(input)


def month_to_int(month):
    switcher = {
        "Jan":0,
        "Feb":1,
        "Mar":2,
        "Apr":3,
        "May":4,
        "June":5,
        "Jul":6,
        "Aug":7,
        "Sep":8,
        "Oct":9,
        "Nov":10,
        "Dec":11
    }

    return switcher.get(month)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    return model.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = 0
    specificity = 0
    for i in range(len(predictions)):
        if labels[i] == 1 and predictions[i] == 1:
            sensitivity += 1
        elif labels[i] == 0 and predictions[i] == 0:
            specificity += 1

    return ((sensitivity / labels.count(1)), (specificity / labels.count(0)))


if __name__ == "__main__":
    main()
