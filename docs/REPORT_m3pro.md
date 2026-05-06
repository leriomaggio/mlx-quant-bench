# MLX Quantization Benchmark — Mistral 7B Instruct v0.3 on M3 Pro

> Real benchmark output from a Mac M3 Pro with 36 GB unified memory, running mlx-quant-bench's `mistral7b-prebuilt` preset 
> (Plan A: 4-bit and 8-bit pre-published variants).

## Summary

| Model | Source | Peak memory | Load time | Avg TTFT | Avg tok/s | Prompts |
|---|---|---|---|---|---|---|
| **8bit** | `mlx-community/Mistral-7B-Instruct-v0.3-8bit` | 7.86 GB | 1.9s | 0.36s | 15.2 | 10 |
| **4bit** | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | 4.29 GB | 0.9s | 0.24s | 27.4 | 10 |

## Memory comparison

Baseline: **8bit** at 7.86 GB (largest of the compared models).

- **4bit**: 4.29 GB (1.83× smaller than 8bit)

## Side-by-side outputs

### `capital_france` (factual)

**8bit** (11.5 tok/s, 8 tokens):

```
The capital of France is Paris.
```

**4bit** (18.2 tok/s, 8 tokens):

```
The capital of France is Paris.
```

### `speed_of_light` (factual)

**8bit** (16.5 tok/s, 51 tokens):

```
The speed of light in a vacuum is approximately 299,792 kilometers per second (km/s). This value is a fundamental constant in physics and is used as a standard for measuring distances and time intervals in the universe.
```

**4bit** (29.5 tok/s, 61 tokens):

```
The speed of light in a vacuum is approximately 299,792 kilometers per second (km/s). This value is often rounded to 300,000 km/s for simplicity. It is a fundamental constant in physics, denoted by the symbol c.
```

### `shakespeare_birth` (factual)

**8bit** (13.2 tok/s, 12 tokens):

```
William Shakespeare was born in 1564.
```

**4bit** (30.3 tok/s, 86 tokens):

```
William Shakespeare was born in the year 1564. He is widely regarded as one of the greatest writers in the English language and is best known for his plays, including "Romeo and Juliet," "Hamlet," "Othello," "Macbeth," and many more. Shakespeare's works have had a profound impact on literature and continue to be widely performed and studied today.
```

### `train_problem` (reasoning)

**8bit** (17.1 tok/s, 137 tokens):

```
Let's reason through this problem:

1. The two trains are moving towards each other, so their combined speed is the sum of their individual speeds: 60 km/h + 90 km/h = 150 km/h.

2. They start 300 km apart. To find the time it takes for them to meet, we can use the formula: Time = Distance / Speed.

3. In this case, Time = 300 km / 150 km/h = 2 hours.

So, it takes 2 hours for the two trains to meet.
```

**4bit** (30.9 tok/s, 163 tokens):

```
Let's reason through this problem:

1. The two trains are initially 300 km apart.
2. They are moving towards each other, so their combined speed is the sum of their individual speeds: 60 km/h + 90 km/h = 150 km/h (combined speed).
3. To find the time it takes for them to meet, we can use the formula Time = Distance / Speed. In this case, the distance is the initial separation between the trains, which is 300 km.
4. So, Time = 300 km / 150 km/h = 2 hours.

Therefore, it takes 2 hours for the two trains to meet.
```

### `age_puzzle` (reasoning)

**8bit** (17.3 tok/s, 187 tokens):

```
Let's denote Bob's current age as x. Since Alice is twice as old as Bob, her age is 2x.

In 5 years, their ages will be (x + 5) for Bob and (2x + 5) for Alice.

According to the problem, the sum of their ages in 5 years will be 40, so we can write the equation:

x + 2x + 5 = 40
3x + 5 = 40

Now, let's solve for x:

3x = 35
x = 35 / 3

Since the age cannot be a fraction, we round down to the nearest whole number, which is 11.5 rounded down to 11.

So, Bob is currently 11 years old.
```

**4bit** (30.9 tok/s, 171 tokens):

```
Let's denote Bob's current age as x (in years). Since Alice is twice as old as Bob, we can denote her age as 2x.

In 5 years, Bob's age will be x + 5, and Alice's age will be 2x + 5.

According to the problem, the sum of their ages in 5 years will be 40:

x + 5 + 2x + 5 = 40
3x + 10 = 40

Now, let's solve for x:

3x = 30
x = 10

So, Bob is currently 10 years old. Alice is 2 * 10 = 20 years old.
```

### `probability` (reasoning)

**8bit** (17.4 tok/s, 237 tokens):

```
To find the probability, we need to determine the number of favorable outcomes (sum equals 7) and divide it by the total number of possible outcomes when rolling two six-sided dice.

First, let's find the total number of possible outcomes. Since each die has 6 sides, when rolling two dice, there are 6*6 = 36 possible outcomes.

Now, let's find the number of favorable outcomes (sum equals 7). The pairs that sum up to 7 are (1,6), (2,5), (3,4), (4,3), (5,2), and (6,1). There are 6 favorable outcomes.

Finally, we can calculate the probability by dividing the number of favorable outcomes by the total number of possible outcomes:

Probability = Favorable outcomes / Total outcomes
Probability = 6 / 36
Probability = 1/6

So, the probability of rolling a sum of 7 with two fair six-sided dice is 1/6.
```

**4bit** (31.6 tok/s, 217 tokens):

```
To find the probability, we need to count the number of favorable outcomes (sum equals 7) and divide it by the total possible outcomes.

Each die has 6 sides, so there are 6*6 = 36 possible outcomes when rolling two dice.

Now, let's find the favorable outcomes where the sum is exactly 7:

1. (1, 6)
2. (2, 5)
3. (3, 4)
4. (4, 3)
5. (5, 2)
6. (6, 1)

There are 6 favorable outcomes. So the probability is the number of favorable outcomes divided by the total possible outcomes:

Probability = Favorable outcomes / Total possible outcomes
= 6 / 36
= 1/6

So the probability that the sum of two rolls of a fair six-sided die is exactly 7 is 1/6.
```

### `json_extract` (instruction)

**8bit** (12.4 tok/s, 15 tokens):

```
{"name": "Alice", "age": "32"}
```

**4bit** (22.5 tok/s, 20 tokens):

```
{
 "name": "Alice",
 "age": "32"
}
```

### `bullet_summary` (instruction)

**8bit** (16.1 tok/s, 83 tokens):

```
1. The Industrial Revolution (18th-19th centuries) revolutionized manufacturing globally, originating in Britain.
2. Key features included the use of steam power, mechanized textile production, and the emergence of the factory system.
3. This period marked a significant shift from agrarian societies to urban industrial centers, leading to rapid economic growth and social change.
```

**4bit** (28.6 tok/s, 93 tokens):

```
- The Industrial Revolution, spanning the 18th and 19th centuries, revolutionized manufacturing globally, with Britain serving as the initial epicenter.
- Key defining features included the widespread use of steam power, the development of mechanized textile production, and the emergence of the factory system.
- This period marked a significant shift from agrarian and handcraft economies to large-scale, mechanized industrial production.
```

### `haiku_format` (instruction)

**8bit** (14.3 tok/s, 24 tokens):

```
Pixels dance in light,

Binary whispers in the dark,

World quantized, complete.
```

**4bit** (23.1 tok/s, 22 tokens):

```
Pixels of light,

Boundaries of the unseen,
Quantum dance repeats.
```

### `negation` (instruction)

**8bit** (16.5 tok/s, 55 tokens):

```
1. Jellyfish

2. Octopus

3. Platypus (Although it might be confusing, the platypus is not a mammal but a monotreme, a type of egg-laying mammal)
```

**4bit** (28.5 tok/s, 45 tokens):

```
1. Jellyfish

2. Octopus

3. Spider (although some species are often referred to as "spider-like arachnids," they are not mammals)
```
