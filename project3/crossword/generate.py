from os import O_WRONLY
import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # List to keep track of the words to remove
        words_to_remove = []

        # Loop through all variables and words in the variable
        for variable, words in self.domains.items():
            for word in words:
                if len(word) != variable.length:
                    words_to_remove.append(word)
            
            # Removes the words with more or less characters than variable.length
            if len(words_to_remove) > 0:
                for word in words_to_remove:
                    self.domains[variable].remove(word)
            words_to_remove.clear()


    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        # List to keep track of the words to remove
        words_to_remove = []

        # Gets the overlap between variables x and y
        intersection = self.crossword.overlaps[x, y]

        if intersection is None:
            return False
        else:
            # If a word in x domain has a correspondence in y domain, keep the word in x domain
            for wordx in self.domains[x]:
                flag = False
                for wordy in self.domains[y]:
                    if wordx[intersection[0]] == wordy[intersection[1]]:
                        flag = True
                        continue
                if not flag:
                    words_to_remove.append(wordx)
        
        if len(words_to_remove) > 0:
            for word in words_to_remove:
                self.domains[x].remove(word)
            return True
        else:
            return False


    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # List of arcs
        arcs_list = []

        # Check whether arcs is None in order to fulfill arcs_list
        if arcs is None:
            for var in self.crossword.variables:
                neighbors = self.crossword.neighbors(var)
                for neighbor in neighbors:
                    if (neighbor, var) not in arcs_list:
                        arcs_list.append((var, neighbor))
        else:
            arcs_list = arcs.copy()
        
        while len(arcs_list) != 0:
            # Remove the arcs from the list
            x = arcs_list[0][0]
            y = arcs_list[0][1]
            arcs_list = arcs_list[1:]
            # If revise is true
            if self.revise(x,y):
                # If the resulting domain of x is 0, the csp is unsolvable
                if len(self.domains[x]) == 0:
                    return False
                # Necessary to check if all the arcs associated to x are still consistent
                else:
                    neighbors = self.crossword.neighbors(x)
                    if y in neighbors:
                        neighbors.remove(y)
                    if len(neighbors) > 0:
                        for neighbor in neighbors:
                            if (x, neighbor) not in arcs_list and (neighbor, x) not in arcs_list:
                                arcs_list.append((x, neighbor))
        
        return True


    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # Check if assignment is complete
        if len(assignment) == len(self.crossword.variables):
            return True
        else:
            return False


    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # List to keep track of the words in the assignment
        words_in_assigment = []
        
        # 1) The words must be distinct
        for word in assignment.values():
            if word not in words_in_assigment:
                words_in_assigment.append(word)
            else:
                return False

        # 2) Every word is the correct length
        for var, word in assignment.items():
            if not var.length == len(word):
                return False
        
        # 3) There are no conflicts between neighboring variables
        list_of_variables = []
        if len(assignment) == 1:
            return True
        # Gets the overlap between variables x and y
        else:
            for var in assignment.keys():
                list_of_variables.append(var)
        
        # Check whether there is an intersection and, if so,
        # if the two words have the same letter
        for i in range(0, len(list_of_variables)):
            for j in range(i + 1, len(list_of_variables)):
                x = list_of_variables[i]
                y = list_of_variables[j]
                intersection = self.crossword.overlaps[x, y]
                if intersection is not None:
                    wordx = assignment[x]
                    wordy = assignment[y]
                    if wordx[intersection[0]] != wordy[intersection[1]]:
                        return False
        
        return True


    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # List of words in the domain of 'var' that shall be returned
        list_of_words = []

        if len(self.domains[var]) == 1:
            list_of_words = [word for word in self.domains[var]]
            return list_of_words
        
        # Dictionary to keep track of how many words in var each word eliminates
        words_eliminated = {word:0 for word in self.domains[var]}

        # List of all variables already defined
        var_in_assignment = [var_assigned for var_assigned in assignment.keys()]

        # Get the number of words in other variables each word in var will eliminate
        for random_var in self.crossword.variables:
            if random_var != var and random_var not in var_in_assignment:
                intersection = self.crossword.overlaps[var, random_var]
                if intersection is not None:
                    for wordx in self.domains[var]:
                        for wordy in self.domains[random_var]:
                            if wordx[intersection[0]] != wordy[intersection[1]]:
                                words_eliminated[wordx] += 1
        
        words_eliminated_sorted = dict(sorted(words_eliminated.items(), key=lambda item: item[1]))
        list_of_values = [word for word in words_eliminated_sorted.keys()]
        
        return list_of_values


    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # List of all variables already defined
        var_in_assignment = [var_assigned for var_assigned in assignment.keys()]

        # Dictionary to keep track of how many remaining values each variable has
        words_in_var = {var:0 for var in self.crossword.variables if var not in var_in_assignment}

        # Get the variable with the minimum number of remaining values
        for random_var in self.crossword.variables:
            if random_var not in var_in_assignment:
                words_in_var[random_var] = len(self.domains[random_var])
        
        words_in_var_sorted = dict(sorted(words_in_var.items(), key=lambda item: item[1]))

        # Check whether there is a tie in the number of variables with less words
        list_of_variables = []
        less_words = list(words_in_var_sorted.values())[0]
        for var, n_words in words_in_var_sorted.items():
            if n_words == less_words:
                list_of_variables.append(var)
        
        # If only one variable has the minimum number of remaining words, this variable will be returned
        if len(list_of_variables) == 1:
            return list_of_variables[0]
        # Else, if there is a tie, the variable with the highest degree will be returned
        else:
            largest_degree = {var:0 for var in list_of_variables}
            for var_n in list_of_variables:
                largest_degree[var_n] = len(self.crossword.neighbors(var_n))
            largest_degree_sorted = dict(sorted(largest_degree.items(), key=lambda item: item[1]))
            return list(largest_degree_sorted.keys())[-1]


    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
         # Check if assignment is complete
        if self.assignment_complete(assignment):
            return assignment
        
        # Try a new variable
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value
            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result
        
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
