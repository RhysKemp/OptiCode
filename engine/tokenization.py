from io import BytesIO
import tokenize

# Class is not language agnostic and as such if we were to implement this into other languages we would need to create an interface and other concrete implementations of the class
# Remains to be seen if the class is needed at all.

def tokenize_code(source_code):
    """
    Tokenizes the given source code into a list of tokens.

    Args:
        source_code (str): The source code to tokenize.

    Returns:
        list: A list of tuples, where each tuple contains the following elements:
            - token.type (int): The type of the token.
            - token.string (str): The string representation of the token.
            - token.start (tuple): The starting (row, column) position of the token.
            - token.end (tuple): The ending (row, column) position of the token.
            - token.line (str): The entire line of code where the token is located.

    Raises:
        tokenize.TokenError: If there is an error during tokenization.
    """
    tokens = []
    try:
        token_stream = tokenize.tokenize(BytesIO(source_code.encode('utf-8')).readline)
        for token in token_stream:
            if token.type != tokenize.ENCODING: # Ignore encoding tokens
                tokens.append((token.type, token.string, token.start, token.end, token.line))
    except tokenize.TokenError as e:
        print(f'Tokenization Error: {e}')
    return tokens

if __name__ == "__main__":
    # Simple display of function
    sample_code = "x = 10\nprint(x)"
    print(tokenize_code(sample_code))