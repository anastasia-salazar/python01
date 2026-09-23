import re


class Tokenizer:
    def __init__(self):
        # The vocabulary: maps each unique word to its integer ID
        self.vocab = {}
        self.next_id = 1  # IDs start at 1, as in the assignment example

    def tokenize(self, text):
        """Split text into lowercase word tokens, dropping punctuation."""
        # [a-z0-9']+ matches runs of letters, digits and apostrophes,
        # so "test." -> "test" and "don't" stays one word
        return re.findall(r"[a-z0-9']+", text.lower())

    def add_word(self, word):
        """Return the word's ID, adding it to the vocabulary if it's new."""
        if word not in self.vocab:
            self.vocab[word] = self.next_id
            self.next_id += 1
        return self.vocab[word]

    def encode(self, text):
        """Tokenize text, update the vocabulary, and return the token IDs."""
        tokens = self.tokenize(text)
        return [self.add_word(token) for token in tokens]


def show(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    print(f"Input:      {text}")
    print(f"Tokens:     {tokens}")
    print(f"IDs:        {ids}")
    print(f"Vocabulary: {tokenizer.vocab}\n")


tokenizer = Tokenizer()

# Example from the assignment
show(tokenizer, "This is a test. This test is simple.")

# New text: known words reuse their IDs, new words get the next ID
show(tokenizer, "Is this a new test? Yes, it is new!")

# Let the user type their own sentences; the same vocabulary keeps growing
print("Type a sentence to tokenize (press Enter on an empty line to quit):")
while True:
    text = input("> ")
    if not text.strip():
        break
    show(tokenizer, text)

print(f"Final vocabulary ({len(tokenizer.vocab)} words): {tokenizer.vocab}")
