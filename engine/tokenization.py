from io import BytesIO
import tokenize

def tokenize_code(source_code):
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
    sample_code = "x = 10\nprint(x)"
    print(tokenize_code(sample_code))