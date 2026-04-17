# PSYCH 420 - Reinforcement Learning

Group project for PSYCH 420 at the University of Waterloo introducing reinforcement 
learning and Q-learning implementation.

## Deliverables

- [Presentation](https://h364lee.github.io/psych420_q_learning/presentation/reinforcement_learning.html)
- [Written Report](https://h364lee.github.io/psych420_q_learning/manuscript/rl_manuscript/rl_manuscript.pdf)

## Setup

- Python 3.12.0

## Clone the repo

```bash
git clone https://github.com/h364lee/psych420_q_learning.git
```

## Virtual environment

1. Go into project directory in terminal
2. Create a virtual environment: `python3 -m venv venv`
3. Activate venv: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Run

1. Open `q_learning_maze.ipynb` to run the base Q-learning maze simulation.
2. Open `q_learning_dopamine.ipynb` to run the D1/D2 extension.

## Render presentation

To render in html: `quarto render presentation/reinforcement_learning.qmd`

To preview with live reload: `quarto preview presentation/reinforcement_learning.qmd`

## Authors

Caleb Lee, Catherine Cao, Arshia Mago, Ashlee King, Christina Fan