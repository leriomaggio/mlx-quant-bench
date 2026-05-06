# MLX Quantization Benchmark — Results

## Summary

| Model | Source | Peak memory | Load time | Avg TTFT | Avg tok/s | Prompts |
|---|---|---|---|---|---|---|
| **bf16** | `./models/Mistral-7B-Instruct-v0.3-bf16` | 14.60 GB | 4.0s | 0.66s | 8.5 | 10 |
| **8bit** | `mlx-community/Mistral-7B-Instruct-v0.3-8bit` | 7.86 GB | 1.6s | 0.28s | 15.1 | 10 |
| **4bit** | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | 4.29 GB | 0.9s | 0.23s | 27.0 | 10 |
| **3bit** | `./models/Mistral-7B-Instruct-v0.3-3bit` | 3.40 GB | 0.6s | 0.23s | 32.9 | 10 |

## Memory comparison

Baseline: **bf16** at 14.60 GB (largest of the compared models).

- **8bit**: 7.86 GB (1.86× smaller than bf16)
- **4bit**: 4.29 GB (3.40× smaller than bf16)
- **3bit**: 3.40 GB (4.29× smaller than bf16)

## Side-by-side outputs

### `capital_france` (factual)

**bf16** (6.3 tok/s, 8 tokens):

```
The capital of France is Paris.
```

**8bit** (11.4 tok/s, 8 tokens):

```
The capital of France is Paris.
```

**4bit** (17.9 tok/s, 8 tokens):

```
The capital of France is Paris.
```

**3bit** (20.7 tok/s, 8 tokens):

```
The capital of France is Paris.
```

### `speed_of_light` (factual)

**bf16** (9.0 tok/s, 71 tokens):

```
The speed of light in a vacuum is approximately 299,792 kilometers per second (km/s). This value is a fundamental constant in physics and is used as a standard for measuring distances and time intervals in the universe. It's often rounded to 300,000 km/s for simplicity.
```

**8bit** (16.3 tok/s, 51 tokens):

```
The speed of light in a vacuum is approximately 299,792 kilometers per second (km/s). This value is a fundamental constant in physics and is used as a standard for measuring distances and time intervals in the universe.
```

**4bit** (29.0 tok/s, 61 tokens):

```
The speed of light in a vacuum is approximately 299,792 kilometers per second (km/s). This value is often rounded to 300,000 km/s for simplicity. It is a fundamental constant in physics, denoted by the symbol c.
```

**3bit** (36.1 tok/s, 76 tokens):

```
The speed of light in a vacuum is approximately 299,792 kilometers per second (km/s). This value is often rounded to 300,000 km/s for simplicity. It's important to note that this is a fundamental constant of the universe, and it's one of the most widely used physical constants.
```

### `shakespeare_birth` (factual)

**bf16** (8.9 tok/s, 71 tokens):

```
William Shakespeare was born in 1564. The exact date is April 26, according to the Julian calendar, which was in use in England at the time. Since the Julian calendar was 10 days behind the Gregorian calendar that we use today, his birthday is often celebrated on April 23 in modern times.
```

**8bit** (13.0 tok/s, 12 tokens):

```
William Shakespeare was born in 1564.
```

**4bit** (30.2 tok/s, 86 tokens):

```
William Shakespeare was born in the year 1564. He is widely regarded as one of the greatest writers in the English language and is best known for his plays, including "Romeo and Juliet," "Hamlet," "Othello," "Macbeth," and many more. Shakespeare's works have had a profound impact on literature and continue to be widely performed and studied today.
```

**3bit** (34.1 tok/s, 62 tokens):

```
William Shakespeare was born in 1564, which is in the 16th century. The exact date is not known, but it is traditionally celebrated as April 26, and he is believed to have been born in Stratford-upon-Avon, England.
```

### `train_problem` (reasoning)

**bf16** (9.0 tok/s, 184 tokens):

```
Let's reason through this problem:

1. The trains are initially 300 km apart.
2. Train A is traveling at 60 km/h, and Train B is traveling at 90 km/h.
3. Since they are moving towards each other, their combined speed is the sum of their individual speeds: 60 km/h + 90 km/h = 150 km/h (combined speed).
4. To find out how long it takes for them to meet, we can divide the total distance between them by their combined speed: Time = Distance / Speed.
5. Plugging in the values, we get: Time = 300 km / 150 km/h = 2 hours.

So, it takes 2 hours for the two trains to meet.
```

**8bit** (16.9 tok/s, 137 tokens):

```
Let's reason through this problem:

1. The two trains are moving towards each other, so their combined speed is the sum of their individual speeds: 60 km/h + 90 km/h = 150 km/h.

2. They start 300 km apart. To find the time it takes for them to meet, we can use the formula: Time = Distance / Speed.

3. In this case, Time = 300 km / 150 km/h = 2 hours.

So, it takes 2 hours for the two trains to meet.
```

**4bit** (29.1 tok/s, 163 tokens):

```
Let's reason through this problem:

1. The two trains are initially 300 km apart.
2. They are moving towards each other, so their combined speed is the sum of their individual speeds: 60 km/h + 90 km/h = 150 km/h (combined speed).
3. To find the time it takes for them to meet, we can use the formula Time = Distance / Speed. In this case, the distance is the initial separation between the trains, which is 300 km.
4. So, Time = 300 km / 150 km/h = 2 hours.

Therefore, it takes 2 hours for the two trains to meet.
```

**3bit** (37.5 tok/s, 187 tokens):

```
Here's how we can solve this problem:

1. The two trains are moving towards each other, so their combined speed is the sum of each train's speed: 60 km/h + 90 km/h = 150 km/h.

2. We know the distance between the two stations is 300 km.

3. To find the time it takes for the trains to meet, we can use the formula: time = distance / speed. However, since both trains are moving towards each other, we only need to consider the distance between them, not the total distance from one station to the other. So, the time it takes for them to meet is: time = (300 km) / (150 km/h) = 2 hours.

So, the two trains will meet in 2 hours.
```

### `age_puzzle` (reasoning)

**bf16** (9.1 tok/s, 160 tokens):

```
Let's denote Bob's current age as x. Since Alice is twice as old as Bob, her age is 2x.

In 5 years, their ages will be (x + 5) for Bob and (2x + 5) for Alice.

According to the problem, the sum of their ages in 5 years will be 40, so we can write the equation:

x + 2x + 5 = 40
3x + 5 = 40

Now, let's solve for x:

3x = 39
x = 39 / 3
x = 13

So, Bob is 13 years old now.
```

**8bit** (17.0 tok/s, 187 tokens):

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

**4bit** (30.7 tok/s, 171 tokens):

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

**3bit** (37.6 tok/s, 200 tokens):

```
Let's denote Bob's current age as B and Alice's current age as A.

We know that Alice is twice as old as Bob, so:

A = 2B

In 5 years, the sum of their ages will be 40. So, in 5 years:

(A + 5) + (B + 5) = 40
(A + B) + 10 = 40

Since A = 2B, we can substitute A in the equation above:

(2B + B) + 10 = 40
3B + 10 = 40

To solve for B, subtract 10 from both sides:

3B = 30

Divide both sides by 3:

B = 10

So, Bob is 10 years old now.
```

### `probability` (reasoning)

**bf16** (9.2 tok/s, 240 tokens):

```
To find the probability, we need to determine the number of favorable outcomes (sum equals 7) and divide it by the total possible outcomes.

1. First, let's find the total possible outcomes. Since we are rolling two six-sided dice, there are 6 sides on each die, so there are 6 * 6 = 36 possible outcomes.

2. Now, let's find the number of favorable outcomes (sum equals 7). The pairs that sum up to 7 are (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), and (6, 1). There are 6 favorable outcomes.

3. Finally, we calculate the probability by dividing the number of favorable outcomes by the total possible outcomes:

   Probability = Favorable outcomes / Total possible outcomes
   = 6 / 36
   = 1/6

So, the probability of rolling a sum of 7 with two fair six-sided dice is 1/6.
```

**8bit** (17.1 tok/s, 237 tokens):

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

**4bit** (31.1 tok/s, 217 tokens):

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

**3bit** (38.7 tok/s, 257 tokens):

```
To find the probability of rolling a sum of 7 with two six-sided dice, we can use the concept of combinations and the multiplication principle.

1. First, we need to determine the total number of outcomes when rolling two dice. A six-sided die has 6 sides, so when rolling two dice, there are 6*6 = 36 possible outcomes.

2. Next, we need to find the number of outcomes that result in a sum of 7. These are the combinations of dice rolls that sum to 7: (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), and (6, 1). There are 6 such outcomes.

3. The probability of an event is the ratio of the number of favorable outcomes to the total number of outcomes. In this case, the probability of rolling a sum of 7 is the ratio of the number of favorable outcomes (6) to the total number of outcomes (36).

So, the probability of rolling a sum of 7 with two six-sided dice is 6/36 = 1/6.
```

### `json_extract` (instruction)

**bf16** (7.4 tok/s, 15 tokens):

```
{"name": "Alice", "age": "32"}
```

**8bit** (12.6 tok/s, 15 tokens):

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

**3bit** (23.0 tok/s, 15 tokens):

```
{"name": "Alice", "age": 32}
```

### `bullet_summary` (instruction)

**bf16** (8.9 tok/s, 92 tokens):

```
- The Industrial Revolution (18th-19th centuries) revolutionized manufacturing, originating in Britain and subsequently spreading globally.
- Key elements included the adoption of steam power, mechanization of textile production, and the emergence of the factory system, which increased productivity and efficiency.
- This period marked a significant shift from agrarian societies to urban industrial centers, leading to profound social, economic, and cultural changes worldwide.
```

**8bit** (16.2 tok/s, 83 tokens):

```
1. The Industrial Revolution (18th-19th centuries) revolutionized manufacturing globally, originating in Britain.
2. Key features included the use of steam power, mechanized textile production, and the emergence of the factory system.
3. This period marked a significant shift from agrarian societies to urban industrial centers, leading to rapid economic growth and social change.
```

**4bit** (28.8 tok/s, 93 tokens):

```
- The Industrial Revolution, spanning the 18th and 19th centuries, revolutionized manufacturing globally, with Britain serving as the initial epicenter.
- Key defining features included the widespread use of steam power, the development of mechanized textile production, and the emergence of the factory system.
- This period marked a significant shift from agrarian and handcraft economies to large-scale, mechanized industrial production.
```

**3bit** (36.1 tok/s, 129 tokens):

```
1. The Industrial Revolution (18th-19th centuries) was a global transformation in manufacturing, originating in Britain, which significantly impacted various industries, particularly textiles.

2. Key defining features of the Industrial Revolution included the widespread use of steam power, which enabled machines to perform repetitive tasks more efficiently, and the factory system, which concentrated workers and production processes in one location.

3. The Industrial Revolution had far-reaching effects, including the urbanization of cities, the emergence of a new working class, and the globalization of trade, as the innovations spread worldwide.
```

### `haiku_format` (instruction)

**bf16** (8.0 tok/s, 23 tokens):

```
Pixels dance in light,

Binary whispers in space,

World quantized, complete.
```

**8bit** (14.4 tok/s, 24 tokens):

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

**3bit** (28.5 tok/s, 26 tokens):

```
In bits, it leaps,

Silent dance, endless,

Quantum world, unveiled.
```

### `negation` (instruction)

**bf16** (8.9 tok/s, 65 tokens):

```
1. Jellyfish

2. Octopus

3. Platypus (Although it may seem like a mammal due to its fur and laying eggs, it is actually a monotreme, a type of mammal that lays eggs instead of giving live birth to young)
```

**8bit** (16.1 tok/s, 55 tokens):

```
1. Jellyfish

2. Octopus

3. Platypus (Although it might be confusing, the platypus is not a mammal but a monotreme, a type of egg-laying mammal)
```

**4bit** (27.7 tok/s, 45 tokens):

```
1. Jellyfish

2. Octopus

3. Spider (although some species are often referred to as "spider-like arachnids," they are not mammals)
```

**3bit** (36.5 tok/s, 85 tokens):

```
1. Jellyfish

2. Platypus (Though it's a mammal, it's an exception because it lays eggs like a reptile and has a duck-like bill)

3. Cnidaria (A group of aquatic animals that include corals and jellyfish, which have no backbone, brain, or digestive system)
```
