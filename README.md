# New Chad Neural
Welcome to the New Chad Neural lib project! As the name suggests this is a library to train neural networks, made by Chads (there is really only me).

## Background
Back in April 2023, a friend and I started digging into AI and machine learning and thought that it was only algebra after all and that it shouldn't be that hard to code a neural network training library as we were pretty good at writing code
AND we had algebra courses in our preparatory class. This is how ***Chad Neural*** was born. But it ended pretty quickly as we weren't algebra Gods (I still am not) and it was basically just a C++ Matrix library.

But then, a year and a half later, I started taking """"deep learning"""" courses (they were really only covering the basics) and I thought that it would be really cool if I coded a 2D game and then coded an AI to beat this 2D game.
And because I wanted this project to be a little bit challenging and not use Python like we learned in class, I thought of the hardest programming language I know, and it was C. So here we go, ***New Chad Neural***.

## Coding philosphy
As I am writting this library in C:
- I have to manage memory myself
- but it is blazing fast

So I wanted something that has a little bit of abstraction, so that "exposed" functions aren't a pain to understand (the ones that the user should use) but not too much abstraction because we are programming in C, not in C++ or in, God forbids, JavaScript.
Based on this, I created some structs like Matrix or Vector but I did not create a function for every single operation with these math objects so Matrix dot product is made by hand everywhere it is needed (this allows for optimizations when possible).
Also, because of this abstraction, almost all arrays, at least the ones that the user isn't aware of, are allocated on the heap. I don't know if this is a problem?

PS: it should be noted that I am not an expert in C nor in machine learning.

## Usage
**TODO**
