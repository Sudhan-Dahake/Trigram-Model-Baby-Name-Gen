import torch


words = open("names.txt", "r").read().splitlines()



# Creates a sorted list of characters (from a to z)
chars = sorted(list(set(''.join(words))))

# maps chars to indexes, a start at 1
stoi = {s:i+1 for i, s in enumerate(chars)}

stoi['.'] = 0

# inverse of stoi.
# maps integers (indexes) to characters.
itos = {i:s for s, i in stoi.items()}




xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']

    for ch1, ch2, ch3 in zip(chs[:], chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]

        # Creating a tuple of two characters
        # xs is the input to the neural net layer before one_hot encoding
        xs.append((ix1, ix2))

        # ys is the target.
        # That is what comes after those two characters present as a tuple in xs.
        ys.append(ix3)


# Converting xs input into a tensor.
# xs shape is batchSize, 2, 27
# batchsize will depend on the number of pairs of names that can be made from our data.
# 2 because trigram model takes 2 inputs
# 27 because each character is one_hot encoded.
xs = torch.tensor(xs)
ys = torch.tensor(ys)


# Generator object
g = torch.Generator().manual_seed(2147483647)

# Weights associated with the first (and only) layer of the neural net.
# 27*2 because each input in xs (after one_hot encoding) contains two lists, each one having 27 items. 
# So each element will have a weight associated with the corresponding neuron.
W = torch.randn((27*2, 27), generator=g, requires_grad=True)


# To get the probabilities between 0 and 1.
# Basically is the exponentiation of logits divided by sum of all the exponentiated logits (normalized)
def softmax(logits):
    return (logits.exp() / ((logits.exp()).sum(1, keepdims=True)))




# Gradient based descent
for k in range(100):
    # forward pass
    xenc = torch.nn.functional.one_hot(xs, num_classes=27).float()

    # -1 is a placeholder to automatically infer that dimesion based on the other dimension
    logits = (xenc.view(-1, W.shape[0])) @ W

    probs = softmax(logits)

    # Negative log likelihood loss function
    loss = -probs[torch.arange(ys.shape[0]), ys].log().mean()

    # This was just showing the loss getting decreased in real time
    # At the end of the iteration loss's value reached around 2.383 (Not bad for a Single layer linear neural net :))
    # print(loss.item())



    # backward pass
    W.grad = None

    loss.backward()



    # update
    # We can afford a higher learning rate here.
    W.data += -10 * W.grad





# Sampling from neural net
for i in range(20):
    out = []

    # The index of the start/end character '.'
    ix = 0  

    while True:
        ch1, ch2 = out[-2] if len(out) > 1 else '.', '.' if len(out) < 1 else out[-1]

        xenc = torch.nn.functional.one_hot(torch.tensor([(stoi[ch1], stoi[ch2])]), num_classes=27).float()

        logits = (xenc.view(-1, W.shape[0])) @ W

        # probabilities for next character
        prob = softmax(logits)



        # This returns the index of the character which has some of the high probabilities.
        ix = torch.multinomial(prob, num_samples=1, replacement=True, generator=g).item()

        out.append(itos[ix])

        if (ix == 0):
            break
        


    print(''.join(out))
