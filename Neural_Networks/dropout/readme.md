# Inverted Dropout

## 1. What is Dropout?

**Dropout** is a regularization technique used during training to reduce **overfitting**.

The idea is simple:

> Randomly turn off some values during each training step.

If `p = 0.5`, each element has a **50% chance of being dropped**.

```text
Input:      [1, 2, 3, 4]

Mask:       [1, 0, 1, 0]

Dropped:    [1, 0, 3, 0]
```

The mask changes randomly each time.

---

## 2. What does `p` mean?

`p` = **probability of dropping an element**.

| `p`    | Drop | Keep |
| ------ | ---: | ---: |
| `0.0`  |   0% | 100% |
| `0.25` |  25% |  75% |
| `0.5`  |  50% |  50% |
| `0.75` |  75% |  25% |

The probability of keeping an element is:

```text
keep probability = 1 - p
```

---

## 3. Convert the Input

```python
x = np.asarray(x)
```

Converts the input into a NumPy array.

This allows us to perform the dropout operation efficiently using **vectorized operations**, without Python loops.

---

## 4. Generate Random Values

```python
if rng is None:
    random_values = np.random.random(x.shape)
else:
    random_values = rng.random(x.shape)
```

We generate **one random number for every element** of `x`.

For example:

```text
x:
[[1, 2],
 [3, 4]]

random_values:
[[0.68, 0.29],
 [0.91, 0.37]]
```

Each value is between `0` and `1`.

---

## 5. Create the Mask

```python
mask = (random_values >= p).astype(float)
```

For `p = 0.5`:

```text
random value >= 0.5 → keep → 1
random value <  0.5 → drop → 0
```

Example:

```text
random_values:
[[0.68, 0.29],
 [0.91, 0.37]]

mask:
[[1, 0],
 [1, 0]]
```

So:

* `1` = keep
* `0` = drop

### Why `.astype(float)`?

The comparison initially produces Boolean values:

```text
True  False
True  False
```

`.astype(float)` converts them to:

```text
1.0  0.0
1.0  0.0
```

---

## 6. Inverted Dropout Scaling

If we simply drop values, the average output becomes smaller.

So we scale the values that survive:

```python
dropout_pattern = mask / (1 - p)
```

For `p = 0.5`:

```text
1 / (1 - 0.5) = 2
```

Therefore:

```text
mask:
[[1, 0],
 [1, 0]]

scaled mask:
[[2, 0],
 [2, 0]]
```

This is why it is called **inverted dropout**.

---

## 7. Apply the Dropout

```python
output = x * dropout_pattern
```

Example:

```text
x:
[[1, 2],
 [3, 4]]

dropout_pattern:
[[2, 0],
 [2, 0]]

output:
[[2, 0],
 [6, 0]]
```

The retained values are multiplied by `2`.

---

## 8. Why Do We Scale?

Suppose:

```text
p = 0.5
```

An element has:

* 50% chance of becoming `0`
* 50% chance of becoming `2x`

For an input value `x`:

```text
Expected output
= 0.5 × 0 + 0.5 × 2x
= x
```

So the expected magnitude stays approximately the same.

**This is the main purpose of inverted dropout scaling.**

---

## 9. Why Return `dropout_pattern`?

The function returns:

```python
return output, dropout_pattern
```

The `dropout_pattern` is the **scaled mask actually applied to the input**.

For `p = 0.5`, it contains:

```text
0 → dropped
2 → retained
```

So:

```python
output = x * dropout_pattern
```

---

## 10. The Complete Flow

```text
Input x
   │
   ▼
Generate random values
   │
   ▼
Compare with p
   │
   ▼
Create mask (0 or 1)
   │
   ▼
Scale mask by 1/(1-p)
   │
   ▼
dropout_pattern
   │
   ▼
x × dropout_pattern
   │
   ▼
Output
```

---

## 11. Training vs Inference

**Dropout is used during training.**

During inference/testing, dropout is normally turned off.

With **inverted dropout**, we already performed the scaling during training, so we don't need additional scaling during inference.

```text
Training:
randomly drop + scale survivors

Inference:
use the complete network normally
```

---

## Key Takeaway

> **Dropout randomly removes elements during training, and inverted dropout scales the surviving elements by `1/(1-p)` so the expected activation magnitude stays unchanged.**

### Remember these 3 lines

```python
mask = (random_values >= p).astype(float)

dropout_pattern = mask / (1 - p)

output = x * dropout_pattern
```

**Mask → Scale → Apply.**
