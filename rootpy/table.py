import sys

def get_max_width(table, index):
    """
    Get the maximum width of the given column index
    """
    return max([len(row[index]) for row in table])

def pprint_table(table, stream = None):
    """
    Prints out a table of data, padded for alignment
    @param stream: Output stream (file-like object)
    @param table: The table to print. A list of lists.
    Each row must have the same number of columns.
    """
    if stream is None:
        stream = sys.stdout

    col_paddings = []

    for i in range(len(table[0])):
        col_paddings.append(get_max_width(table, i))

    for row in table:
        # left col
        print >> out, row[0].ljust(col_paddings[0] + 1),
        # rest of the cols
        for i in range(1, len(row)):
            col = row[i].rjust(col_paddings[i] + 2)
            print >> out, col,
        print >> out
