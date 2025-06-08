import pandas as pd

# Generate a larger, balanced dataset
data = []

# 10 spam samples (label = 1)
for _ in range(10):
    row = [0.2, 0.3, 0.5, 0.0, 0.1, 0.2, 0.0, 0.0, 0.05, 0.1, 0.01, 0.01, 0.0, 0.0, 0.0,
           0.2, 0.1, 0.01, 5.0, 100, 500, 1]
    data.append(row)

# 10 non-spam samples (label = 0)
for _ in range(10):
    row = [0.05, 0.1, 0.15, 0.0, 0.05, 0.1, 0.0, 0.0, 0.02, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.1, 0.05, 0.0, 3.0, 50, 200, 0]
    data.append(row)

columns = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "char_freq_;", "char_freq_(",
    "char_freq_[", "char_freq_!", "char_freq_]", "char_freq_'", "char_freq_$",
    "char_freq_#", "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total", "spam"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("spam.csv", index=False)

print("File 'spam.csv' created with 20 balanced rows.")
