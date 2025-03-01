
## Can GPT find Hidden Patterns?

LLMs can do amazing things like writing, summarizing, and processing data in an unprecedented way. But how well can it detect patterns in our everyday life?

In simple terms: can foundation models like GPT detect patterns in unknown data (for example, housing prices or sales figures) without additional training or fine-tuning?

Could it, for instance, tell a salesperson which products should be sold less and which should be sold more based on the data?
>  There are lots of great tools and techniques for training models to predict data, and some can cluster your data. But you need to know how. 
Wouldn’t it be great if the LLM that everyone already uses could help with that, at least to get a glimpse of hidden patterns you might never have considered?

We did a test with three popular foundation models: **OpenAI o3-mini**, **Claude Sonnet 3.5**, and **Gemini**.

And here is what I did:

 1. I generated an algorithm based on a set of “hidden” pattern (basically just an algorithm) so we can easily judge the analysis results using the code as the source of truth.

 2. I used a typical example for housing prices (but with some hidden patterns, as you’ll see).

 3. I generated a dataset with this code.

 4. I asked the foundation models (GPT, Claude, Gemini) to find hidden patterns without any further instructions.

 5. I took the response and asked a separate judge to check the correctness of the results, based on the algorithm defined earlier.
>  **Disclaimer:** For large datasets and in enterprise settings, you would likely use special models and training algorithms. This experiment is mainly for exploration, not for professional use.

Having said that, let’s dive in.

— -

### Generate the Code

Here are the patterns we have built into the generated datasets for 1000 entries of house prices.
>  Note that the LLMs might find other regularities in the dataset. The goal of this exercise is not to expect the LLMs to find exactly these patterns (though that would be interesting) but to check whether our assumptions about the data are correct. The aim is to be aware of correlations we might have missed if we hadn’t applied any form of intelligence to our data.

![](https://cdn-images-1.medium.com/max/NaN/1*gmXCv7HG6ebhFuXS-hB61Q.png)

And here is the generated code that implements these patterns.
>  Note that the LLM will never know that such code exists, it will only be aware of the generated data.

![](https://cdn-images-1.medium.com/max/NaN/1*he9netd-lTrnG1PFzwNj_g.png)

Let’s see how the LLMs interpret the data they generate and what general assumptions they make about the data.

— -

### GPT o3 mini

We start by giving the model the generated dataset (produced by running the code above, which creates 1000 items). Then we simply ask the AI (o3 mini in this case) to explain the data.

![](https://cdn-images-1.medium.com/max/NaN/1*zqR0Vum_lMNQHnYN-nuMhA.png)

**The answer (cut-off):**

![](https://cdn-images-1.medium.com/max/NaN/1*E3LskWPM6t6Vjhxhu9ImHg.png)

### Evaluation by the LLM judge

As described, we took the answer and gave it to the LLM judge (o3-mini) and here is what it says:

![](https://cdn-images-1.medium.com/max/3080/1*qGjWzCQHnGiEIdwmt20obQ.png)
>  Please note that the prompt is quite simple and can be tweaked to focus more on finding certain patterns or correlations in the data.

— -

### Claude

We did the same experiment with Claude, but for brevity, here are only the results of the evaluation:

**Evaluation:**

![](https://cdn-images-1.medium.com/max/3060/1*5Wg9DaAbQ9673P320igt6A.png)

— -

### Gemini 2.0 Pro

Here is the evaluation of the Gemini analysis:

**Evaluation**

![](https://cdn-images-1.medium.com/max/3072/1*U6VPHoU9mtpL69-dE74oLw.png)

Gemini is obviously much more talkative than its counterparts, and it makes a lot of wrong assumptions.

— -

### Comparison and Overall Recommendation

As you can see, the patterns recognised are very diverse and you might wonder which model performed best for such tasks.

I gave all the evaluations to the judge, and here is the summary:

![](https://cdn-images-1.medium.com/max/3060/1*lWmmdAW0K6yOcFuu0R5cSg.png)

**Overall Recommendation:**
O3-mini demonstrates the best capability for predicting patterns in this dataset, as its analysis aligns most accurately with the underlying data generation process.
[**AI Rabbit News**
*AI News & Tutorials*airabbit.blog](https://airabbit.blog/)

### Wrap-Up

We are probably only aware of a fraction of the patterns and collisions in our everyday lives, partly because of the sheer volume of data and the unexpected correlations we sometimes notice — and sometimes miss.

With this exercise, we challenged foundation models without any training or fine-tuning to look for patterns in a dataset. The results were not too bad.

However, it’s always wise to check any assumptions that LLms make (not just in this kind of experiment), and to try to validate the assumptions yourself or through automated methods. AI may give you a glimpse of something you could never have seen, but you should always question and use common sense as a human before making a decision.

