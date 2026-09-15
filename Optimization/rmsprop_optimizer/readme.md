# RMSProp Optimizer

## What is RMSProp?

**RMSProp** is an optimization algorithm used to update the weights of a machine learning model during training.

Like gradient descent, it uses the **gradient** to decide how to change the weights.

The difference is:

> **RMSProp keeps track of the size of recent gradients and uses that information to control the step size for each weight.**

This can make training more stable.

---

## 1. The main idea

Normal gradient descent does:

```text
weight = weight - learning_rate × gradient
```

The problem is that different parameters can have very different gradient sizes.

For example:

```text
gradient = [0.1, 10.0]
```

Using the same learning rate for both can cause one parameter to move too much.

RMSProp adjusts the update based on the **recent size of the gradients**.

---

## 2. The inputs

Our function receives:

```python
def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
)
```

### `w` — weights

```python
w = [1.0, 2.0]
```

These are the current parameters of the model.

Think:

```text
weight 1 = 1.0
weight 2 = 2.0
```

---

### `g` — gradients

```python
g = [0.2, -0.4]
```

The gradient tells us how each weight should change.

For example:

```text
gradient > 0 → decrease the weight
gradient < 0 → increase the weight
```

---

### `s` — previous moving average

```python
s = [0.0, 0.0]
```

RMSProp keeps a running value representing the **recent squared gradients**.

This allows RMSProp to remember how large the gradients have been.

---

### `lr` — learning rate

```python
lr = 0.1
```

Controls the overall size of the update.

---

### `beta` — decay rate

```python
beta = 0.9
```

Controls how much we remember from the previous `s`.

With:

```text
beta = 0.9
```

we keep:

```text
90% old information
10% new gradient information
```

So `beta` controls the **memory** of RMSProp.

---

### `eps` — small number

```python
eps = 1e-8
```

This prevents division by zero.

It is just a tiny safety value.

---

# 3. Convert lists to NumPy arrays

```python
w = np.array(w)
g = np.array(g)
s = np.array(s)
```

This makes mathematical operations easier.

For example:

```python
g**2
```

squares every element:

```text
[0.2, -0.4]
      ↓
[0.04, 0.16]
```

---

# 4. Update the moving average

This is the most important RMSProp step:

```python
new_s = beta * s + (1 - beta) * g**2
```

The idea is:

> **Combine the previous moving average with the current squared gradient.**

The formula is:

```text
new_s = β × old_s + (1 - β) × gradient²
```

So:

```text
old information + new information
```

---

## Small example

Suppose:

```text
s = [0, 0]
g = [0.2, -0.4]
beta = 0.9
```

First square the gradients:

```text
g² = [0.04, 0.16]
```

Then:

```text
new_s = 0.9 × [0, 0] + 0.1 × [0.04, 0.16]
```

Therefore:

```text
new_s = [0.004, 0.016]
```

So RMSProp now remembers:

```text
The recent gradients had these approximate squared sizes.
```

---

# 5. Update the weights

Now we have:

```python
new_w = w - lr * g / (np.sqrt(new_s) + eps)
```

This is the RMSProp weight update.

Compare it with normal gradient descent:

```text
Gradient Descent:

new_w = w - lr × g
```

RMSProp adds a normalization term:

```text
RMSProp:

new_w = w - lr × g / √new_s
```

The important idea is:

> **Large recent gradients → larger denominator → smaller update.**

> **Small recent gradients → smaller denominator → relatively larger update.**

So RMSProp adapts the step size for each parameter.

---

# 6. Let's calculate one weight

Suppose:

```text
w = 1.0
g = 0.2
new_s = 0.004
lr = 0.1
```

First:

```text
√new_s = √0.004
       ≈ 0.0632
```

Then the update is approximately:

```text
new_w = 1.0 - 0.1 × 0.2 / 0.0632
```

```text
new_w ≈ 1.0 - 0.316
```

```text
new_w ≈ 0.684
```

So:

```text
Before: 1.0
After:  0.684
```

The gradient was positive, so the weight moved downward.

---

# 7. Why square the gradient?

```python
g**2
```

We square the gradient because we want to measure its **size**, not its direction.

For example:

```text
g =  0.5 → g² = 0.25
g = -0.5 → g² = 0.25
```

Both gradients have the same magnitude.

RMSProp cares about:

> **How large are the gradients?**

The direction is still handled separately by `g` in the weight update.

---

# 8. Why `sqrt(new_s)`?

We calculated:

```python
g**2
```

so the values are squared.

Taking the square root brings them back to approximately the original scale:

```text
gradient
   ↓
gradient²
   ↓
moving average
   ↓
√moving average
```

This gives us a value related to the typical recent gradient size.

---

# 9. Why does RMSProp help?

Imagine two weights:

```text
weight 1 → small gradients
weight 2 → large gradients
```

Without RMSProp:

```text
Both use the same learning rate.
```

With RMSProp:

```text
small gradient history → relatively larger step
large gradient history  → relatively smaller step
```

So each parameter can effectively have its own adaptive learning rate.

Think of it like walking downhill:

```text
Normal gradient descent:

Everyone takes steps of similar size.


RMSProp:

Each person adjusts their step size
depending on how steep the terrain has been.
```


---

# Key Takeaways

### Gradient

```text
g
```

Tells us the direction in which the weight should move.

### Squared gradient

```text
g²
```

Tells us about the **size** of the gradient without caring about its direction.

### Moving average

```text
new_s = βs + (1-β)g²
```

Keeps a memory of recent gradient sizes.

### Weight update

```text
new_w = w - lr × g / (√new_s + ε)
```

Uses the gradient but scales the update based on recent gradient sizes.

### `beta`

Controls how much history we remember.

```text
beta high → remember more history
beta low  → focus more on recent gradients
```

### `eps`

Prevents division by zero.

---

## One-sentence summary

> **RMSProp is gradient descent with memory: it remembers the recent size of gradients and uses that information to adapt the step size of each weight.**

```text
Gradient
   ↓
Square it
   ↓
Update moving average (memory)
   ↓
Take square root
   ↓
Normalize gradient
   ↓
Update weights
```
