import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if self.count == len(self.cells):
            return self.cells
        else:
            return None

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells
        else:
            return None

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)
    
    def remove_no_cells(self):
        """
        Removes all sentences from self.knowledge where the number
        of cells in set is equal to 0.
        """
        # Update self.knowledge
        list_to_remove = []
        for existing_sentence in self.knowledge:
            if len(existing_sentence.cells) == 0:
                list_to_remove.append(existing_sentence)
        if len(list_to_remove) > 0:
            for sentence_to_remove in list_to_remove:
                self.knowledge.remove(sentence_to_remove)
        list_to_remove.clear()
    
    def remove_equal_sentences(self):
        """
        Removes duplicated sentences from self.knowledge
        """
        list_to_verify = []
        list_to_verify = self.knowledge.copy()
        list_to_remove = []
        list_to_keep = []
        while len(list_to_verify) != 0:
            sentence_to_verify = list_to_verify.pop()
            if sentence_to_verify not in list_to_remove:
                list_to_keep.append(sentence_to_verify)
            for sentence_to_check in list_to_verify:
                if sentence_to_check == sentence_to_verify:
                    list_to_remove.append(sentence_to_check)
        if len(list_to_remove) > 0:
            for sentence_to_remove in list_to_remove:
                self.knowledge.remove(sentence_to_remove)
        list_to_verify.clear()
        list_to_remove.clear()
        list_to_keep.clear()
    
    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # 1) mark the cell as a move that has been made
        self.moves_made.add(cell)

        # 2) mark the cell as safe
        self.mark_safe(cell)

        # 3) add a new sentence to the AI's knowledge base
        # Create a new set
        new_set = set()
        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                # Ignore the cell itself
                if (i, j) == cell:
                    continue
                # If it is, for sure, a mine cell, decrease 1 from count
                # and do not add cell to knowledge
                if (i,j) in self.mines:
                    count -= 1
                    continue
                # If it is, for sure, a safe cell, ignore
                if (i,j) in self.safes:
                    continue
                # Check if cell is in bounds
                if 0 <= i < self.height and 0 <= j < self.width:
                    new_set.add((i,j))
        # If count is equal to 0, then all neighboring cells are safe
        if count == 0:
            for each_cell in new_set:
                self.mark_safe(each_cell)
        # Otherwise, a new sentence is created
        else:
            new_sentence = Sentence(new_set, count)
            # Ensure that new_sentence is not in self.knowledge yet
            already_exists = False
            for old_sentence in self.knowledge:
                if old_sentence == new_sentence:
                    already_exists = True
                    continue
            if not already_exists:
                self.knowledge.append(new_sentence)
        
        # 4) mark any additional cells as safe or as mines
        for inquiry_sentence in self.knowledge:
            inquiry_safe_set = set()
            inquiry_safe_set = inquiry_sentence.known_safes()
            if inquiry_safe_set is not None:
                for count in range(len(inquiry_safe_set)):
                    inquiry_cell = inquiry_safe_set.pop()
                    inquiry_sentence.mark_safe(inquiry_cell)
                    self.mark_safe(inquiry_cell)
            inquiry_mine_set = set()
            inquiry_mine_set = inquiry_sentence.known_mines()
            if inquiry_mine_set is not None:
                for count in range(len(inquiry_mine_set)):
                    inquiry_cell = inquiry_mine_set.pop()
                    inquiry_sentence.mark_mine(inquiry_cell)
                    self.mark_mine(inquiry_cell)
        
        # Update self.knowledge
        self.remove_no_cells()
        self.remove_equal_sentences()
        
        # 5) add any new sentences to the AI's knowledge base
        all_sentences = []
        all_sentences = self.knowledge.copy()
        sentence_to_write = []
        for x in self.knowledge:
            for y in all_sentences:
                if x == y:
                    pass
                else:
                    if y.cells.issubset(x.cells):
                        inferred_set = set()
                        inferred_set = x.cells.difference(y.cells)
                        inferred_count = x.count - y.count
                        inferred_sentence = Sentence(inferred_set, inferred_count)
                        sentence_to_write.append(inferred_sentence)
        
        if len(sentence_to_write) > 0:
            for z in sentence_to_write:
                if z.count == 0:
                    for each_cell in z.cells:
                        self.mark_safe(each_cell)               
                else:
                    already_exists = False
                    for old_sentence in self.knowledge:
                        if old_sentence == z:
                            already_exists = True
                            continue
                    if not already_exists:
                        self.knowledge.append(z)
        all_sentences.clear()
        sentence_to_write.clear()

        # Update self.knowledge
        self.remove_no_cells()
        self.remove_equal_sentences()
        
    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        safe_moves = self.safes - self.moves_made
        if len(safe_moves) > 0:
            return safe_moves.pop()
        else:
            return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        if (self.height * self.width) == (len(self.moves_made) + len(self.mines)):
            return None
        else:
            while True:
                i = random.randrange(self.height)
                j = random.randrange(self.width)
                if (i,j) not in self.moves_made and (i,j) not in self.mines:
                    return (i,j)