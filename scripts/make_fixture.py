"""Build a synthetic `serve.duckdb` that follows `observatorio.publish_schema`.

The viewer is developed against this fixture until the real pipeline publishes.
Everything is made up but shaped like WildChat-1M: a 3-level hierarchy
(~15 top, ~80 mid, ~600 base clusters) with Clio-style names, siblings placed
near each other on the map, WildChat-like language / country / model mixes,
monthly volumes over 2023-04..2024-04, and a shuffled `map_points` cloud.

Pure CPU, numpy only, deterministic (seed 7).

    cd ~/observatorio && source env.sh
    uv run python scripts/make_fixture.py            # -> $OBS_DATA/fixture/serve.duckdb
    uv run python scripts/make_fixture.py --out /tmp/x.duckdb --base 300
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo isn't an installed package

from observatorio.config import DATA, DATASET, EMBED_MODEL, LLM_MODEL
from observatorio.publish_schema import COLUMNS, create_schema

SEED = 7
MIN_USERS, MIN_CONVERSATIONS = 10, 20
MONTHS = [f"{y}-{m:02d}" for y, m in [(2023, m) for m in range(4, 13)] + [(2024, m) for m in range(1, 5)]]
# Rough WildChat volume by month (relative).
MONTH_W = np.array([6.0, 8.0, 8.5, 8.0, 7.5, 7.0, 7.5, 8.0, 8.0, 8.0, 8.5, 8.0, 5.0])
EARLY, LATE = slice(0, 6), slice(7, 13)  # halves of the range; the middle month counts for neither

LANGS = {
    "English": 0.47,
    "Chinese": 0.23,
    "Russian": 0.12,
    "French": 0.03,
    "Spanish": 0.025,
    "Portuguese": 0.02,
    "German": 0.02,
    "Arabic": 0.015,
    "Italian": 0.01,
    "Turkish": 0.01,
    "Japanese": 0.008,
    "Korean": 0.006,
    "Vietnamese": 0.005,
    "Other": 0.021,
}
COUNTRY_GIVEN_LANG = {
    "English": {
        "United States": 0.46,
        "United Kingdom": 0.08,
        "India": 0.08,
        "Canada": 0.05,
        "Australia": 0.04,
        "Philippines": 0.03,
        "Germany": 0.02,
        "Other": 0.24,
    },
    "Chinese": {
        "China": 0.62,
        "Hong Kong": 0.14,
        "Taiwan": 0.09,
        "Singapore": 0.05,
        "United States": 0.05,
        "Other": 0.05,
    },
    "Russian": {"Russia": 0.76, "Ukraine": 0.08, "Kazakhstan": 0.06, "Belarus": 0.04, "Other": 0.06},
    "French": {"France": 0.7, "Canada": 0.1, "Belgium": 0.07, "Morocco": 0.05, "Other": 0.08},
    "Spanish": {
        "Spain": 0.3,
        "Mexico": 0.25,
        "Argentina": 0.1,
        "Colombia": 0.1,
        "Chile": 0.08,
        "Other": 0.17,
    },
    "Portuguese": {"Brazil": 0.86, "Portugal": 0.1, "Other": 0.04},
    "German": {"Germany": 0.8, "Austria": 0.08, "Switzerland": 0.08, "Other": 0.04},
    "Arabic": {
        "Egypt": 0.3,
        "Saudi Arabia": 0.25,
        "Morocco": 0.15,
        "United Arab Emirates": 0.1,
        "Other": 0.2,
    },
    "Italian": {"Italy": 0.92, "Other": 0.08},
    "Turkish": {"Turkey": 0.95, "Other": 0.05},
    "Japanese": {"Japan": 0.93, "Other": 0.07},
    "Korean": {"South Korea": 0.93, "Other": 0.07},
    "Vietnamese": {"Vietnam": 0.95, "Other": 0.05},
    "Other": {"Other": 1.0},
}
GPT35 = ["gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-0125"]
GPT4 = ["gpt-4-0314", "gpt-4-1106-preview", "gpt-4-0125-preview"]
MODELS = GPT35 + GPT4


def model_version(month: int, gpt4: bool) -> int:
    """Index of the snapshot that was live in month index `month` (0 = 2023-04)."""
    if gpt4:  # 0314 until Oct 2023, 1106-preview Nov-Jan, 0125-preview from Feb 2024
        return 0 if month < 7 else (1 if month < 10 else 2)
    return 0 if month < 3 else (1 if month < 10 else 2)  # 0301 / 0613 / 0125


TURN_KEYS = ["1", "2", "3-5", "6+"]
TURN_VALUES = np.array([1.0, 2.0, 3.8, 9.5])  # representative mean per bucket

# --------------------------------------------------------------------------------------------
# Taxonomy. Tops are listed so that related tops sit next to each other on the map.
# Each mid: (name, description, verbs, objects, detail sentences for base descriptions).
# Profile knobs: weight (share of conversations), lang (multiplicative tilt), concern (P(1..5)),
# turns (P over TURN_KEYS), gpt4 (share on GPT-4), users (users per conversation), trend, mod.
# --------------------------------------------------------------------------------------------
TAXONOMY = [
    {
        "name": "Write, debug, and explain software code across languages",
        "desc": "Users asked the assistant to write, fix, and explain code in many programming languages and frameworks. Requests ranged from small scripts and error messages to full application features and architecture questions.",
        "weight": 0.19,
        "lang": {"Chinese": 1.3, "Russian": 1.2},
        "concern": [0.9, 0.07, 0.02, 0.008, 0.002],
        "turns": [0.35, 0.2, 0.3, 0.15],
        "gpt4": 0.38,
        "users": 0.7,
        "trend": 0.25,
        "mod": 0.01,
        "mids": [
            (
                "Debug and fix errors in Python scripts",
                "Users shared Python code with error messages and asked for fixes. The conversations covered tracebacks, library misuse, and logic bugs in data and automation scripts.",
                ["Debug", "Fix errors in", "Refactor"],
                [
                    "Python web scraping scripts",
                    "pandas data-cleaning code",
                    "Selenium automation scripts",
                    "asyncio networking code",
                    "file-processing utilities",
                    "Tkinter GUI applications",
                    "Telegram bot handlers",
                    "Discord bot commands",
                    "command-line argument parsers",
                ],
                [
                    "Users pasted tracebacks and asked for corrected versions of their code.",
                    "Follow-up turns usually requested small changes after the first fix.",
                ],
            ),
            (
                "Build and style front-end web interfaces",
                "Users asked for HTML, CSS, and JavaScript to build or restyle web pages. Requests included layouts, animations, and components in popular front-end frameworks.",
                ["Build", "Style", "Fix layout issues in"],
                [
                    "responsive landing pages",
                    "React components with hooks",
                    "Vue single-file components",
                    "CSS grid and flexbox layouts",
                    "navigation menus and modals",
                    "form validation logic",
                    "CSS animations and transitions",
                    "Tailwind-styled dashboards",
                ],
                [
                    "Users often supplied existing markup and asked for a specific visual change.",
                    "Several requests asked for complete, copy-ready files.",
                ],
            ),
            (
                "Develop back-end services and REST APIs",
                "Users asked for help designing and implementing server-side code and APIs. Conversations covered routing, authentication, databases, and deployment configuration.",
                ["Implement", "Debug", "Design"],
                [
                    "Node.js Express REST endpoints",
                    "Spring Boot services",
                    "Django REST framework views",
                    "FastAPI applications",
                    "JWT authentication flows",
                    "WebSocket chat servers",
                    "Go HTTP handlers",
                    "GraphQL resolvers",
                ],
                [
                    "Users typically described the desired behavior and asked for working code.",
                    "Many follow-ups concerned error handling and database access.",
                ],
            ),
            (
                "Write and debug C, C++, and Java programs",
                "Users asked for help with compiled-language programs, often for coursework. Requests covered memory management, data structures, and object-oriented design.",
                ["Write", "Debug", "Explain"],
                [
                    "C++ linked list and tree implementations",
                    "Java object-oriented class hierarchies",
                    "C pointer and memory-allocation bugs",
                    "C++ template code",
                    "Java multithreading programs",
                    "C socket programming exercises",
                    "Java Swing interfaces",
                    "C++ STL algorithm usage",
                ],
                [
                    "Many requests came from programming assignments with stated constraints.",
                    "Users frequently asked for line-by-line explanations of the result.",
                ],
            ),
            (
                "Explain algorithms and data structures",
                "Users asked for explanations and implementations of classic algorithms. Conversations often included complexity analysis and competitive-programming problems.",
                ["Explain", "Implement", "Optimize"],
                [
                    "dynamic programming solutions",
                    "graph traversal algorithms",
                    "sorting algorithm comparisons",
                    "binary search variations",
                    "competitive programming problems",
                    "hash table implementations",
                    "shortest-path algorithms",
                ],
                [
                    "Users asked for both intuition and code in their language of choice.",
                    "Complexity analysis was a common follow-up request.",
                ],
            ),
            (
                "Write scripts for automation and system administration",
                "Users asked for shell scripts and configuration for servers and developer tooling. Requests covered Linux commands, containers, and CI pipelines.",
                ["Write", "Fix", "Explain"],
                [
                    "Bash backup scripts",
                    "Dockerfiles and compose files",
                    "PowerShell automation scripts",
                    "Nginx reverse-proxy configs",
                    "GitHub Actions workflows",
                    "Linux permission and process commands",
                    "Kubernetes deployment manifests",
                ],
                [
                    "Users usually described their environment and the task to automate.",
                    "Several conversations focused on explaining unexpected command output.",
                ],
            ),
            (
                "Build machine learning models and pipelines",
                "Users asked for help training and evaluating machine learning models. Conversations covered PyTorch and TensorFlow code, data preparation, and model tuning.",
                ["Implement", "Debug", "Tune"],
                [
                    "PyTorch training loops",
                    "image classification CNNs",
                    "text classification with transformers",
                    "scikit-learn pipelines",
                    "time-series forecasting models",
                    "reinforcement learning agents",
                    "LSTM sequence models",
                ],
                [
                    "Users often shared shape errors or poor validation metrics.",
                    "Requests commonly asked for complete, runnable training scripts.",
                ],
            ),
            (
                "Develop mobile and desktop applications",
                "Users asked for code for Android, iOS, and desktop apps. Requests included UI screens, local storage, and platform APIs.",
                ["Build", "Debug", "Add features to"],
                [
                    "Android Kotlin activities",
                    "Flutter screens and widgets",
                    "SwiftUI views",
                    "Electron desktop apps",
                    "C# WinForms applications",
                    "Android Room databases",
                ],
                [
                    "Users usually asked for a specific screen or feature to be implemented.",
                    "Follow-ups dealt with build errors and platform permissions.",
                ],
            ),
            (
                "Convert code between programming languages",
                "Users asked for code to be ported from one language or framework to another. Conversations covered idiomatic rewrites and library equivalents.",
                ["Convert", "Port", "Translate"],
                [
                    "Python code to JavaScript",
                    "Java classes to Kotlin",
                    "MATLAB scripts to Python",
                    "jQuery code to vanilla JavaScript",
                    "C# code to Python",
                ],
                [
                    "Users pasted the source code and named the target language.",
                    "Follow-ups fixed behavior differences after the port.",
                ],
            ),
            (
                "Set up development tools and version control",
                "Users asked about configuring editors, package managers, and Git. Requests covered installation errors and repository workflows.",
                ["Configure", "Fix", "Explain"],
                [
                    "Git branching and merge conflicts",
                    "Python virtual environments",
                    "npm and dependency errors",
                    "VS Code extensions and settings",
                    "SSH keys and remote servers",
                ],
                [
                    "Users usually pasted a terminal error.",
                    "Several conversations were step-by-step setup guides.",
                ],
            ),
            (
                "Write smart contracts and blockchain code",
                "Users asked for code interacting with blockchains. Requests covered Solidity contracts, tokens, and wallet integrations.",
                ["Write", "Audit", "Debug"],
                [
                    "Solidity token contracts",
                    "NFT minting contracts",
                    "web3 wallet integrations",
                    "trading bots for crypto exchanges",
                ],
                [
                    "Users often asked for deployable contract code.",
                    "Some requests concerned gas costs and security.",
                ],
            ),
        ],
    },
    {
        "name": "Analyze data and write database queries",
        "desc": "Users asked for help querying databases, analyzing datasets, and building spreadsheets. Requests included SQL, Excel formulas, and statistical summaries.",
        "weight": 0.035,
        "lang": {"Chinese": 1.1},
        "concern": [0.92, 0.06, 0.015, 0.004, 0.001],
        "turns": [0.4, 0.2, 0.28, 0.12],
        "gpt4": 0.33,
        "users": 0.75,
        "trend": 0.15,
        "mod": 0.008,
        "mids": [
            (
                "Write and optimize SQL queries",
                "Users asked for SQL queries against their described schemas. Conversations covered joins, aggregations, window functions, and slow-query tuning.",
                ["Write", "Optimize", "Debug"],
                [
                    "SQL join and aggregation queries",
                    "window-function reports",
                    "MySQL schema designs",
                    "PostgreSQL performance issues",
                    "stored procedures",
                    "SQL Server reporting queries",
                ],
                [
                    "Users typically described tables and the report they needed.",
                    "Some follow-ups asked why a query returned unexpected rows.",
                ],
            ),
            (
                "Build spreadsheet formulas and Excel automation",
                "Users asked for spreadsheet formulas, pivot tables, and macros. Requests focused on business reporting and cleaning tabular data.",
                ["Write", "Fix", "Explain"],
                [
                    "Excel lookup formulas",
                    "VBA macros",
                    "Google Sheets scripts",
                    "pivot table summaries",
                    "conditional formatting rules",
                ],
                [
                    "Users described their columns and the result they wanted.",
                    "Several conversations asked for step-by-step instructions.",
                ],
            ),
            (
                "Perform statistical analysis and visualize data",
                "Users asked for statistical tests, regression models, and charts. Conversations covered interpreting results and writing R or Python analysis code.",
                ["Run", "Interpret", "Visualize"],
                [
                    "regression analyses",
                    "hypothesis tests",
                    "R data-analysis scripts",
                    "matplotlib and seaborn charts",
                    "survey data summaries",
                    "A/B test results",
                ],
                [
                    "Users often supplied summary numbers and asked what they meant.",
                    "Requests commonly ended with a plotting or reporting step.",
                ],
            ),
            (
                "Clean and transform datasets with Python",
                "Users asked for code to load, clean, and reshape tabular data. Conversations centred on pandas and CSV or JSON processing.",
                ["Clean", "Transform", "Merge"],
                [
                    "CSV files with pandas",
                    "JSON API responses",
                    "Excel workbooks with Python",
                    "messy date and text columns",
                    "large log files",
                ],
                [
                    "Users usually described their columns and the target format.",
                    "Follow-ups dealt with performance on larger files.",
                ],
            ),
        ],
    },
    {
        "name": "Design and develop video games and game mods",
        "desc": "Users asked for help building games, from gameplay code to level and mechanic design. Requests spanned Unity, Roblox, Minecraft mods, and browser games.",
        "weight": 0.03,
        "lang": {"Russian": 1.3},
        "concern": [0.85, 0.1, 0.035, 0.01, 0.005],
        "turns": [0.3, 0.2, 0.32, 0.18],
        "gpt4": 0.3,
        "users": 0.6,
        "trend": 0.1,
        "mod": 0.02,
        "mids": [
            (
                "Code game mechanics in Unity and other engines",
                "Users asked for scripts implementing movement, combat, and interaction systems. Most conversations involved Unity C#, Godot, or Unreal blueprints.",
                ["Implement", "Debug", "Improve"],
                [
                    "Unity player controllers",
                    "enemy AI behaviors",
                    "inventory systems",
                    "Godot scene scripts",
                    "Unreal Engine blueprints",
                    "2D platformer physics",
                ],
                [
                    "Users usually described the desired game feel and asked for code.",
                    "Follow-ups dealt with collisions and frame-rate issues.",
                ],
            ),
            (
                "Script Roblox and Minecraft modifications",
                "Users asked for Lua scripts and Java mods for sandbox games. Requests included custom items, commands, and server plugins.",
                ["Write", "Fix", "Extend"],
                [
                    "Roblox Lua game scripts",
                    "Minecraft Forge mods",
                    "Minecraft server plugins",
                    "Roblox GUI scripts",
                    "custom item and crafting recipes",
                ],
                [
                    "Many users were early-stage programmers learning through modding.",
                    "Requests often asked for complete scripts to paste in.",
                ],
            ),
            (
                "Design game concepts, levels, and balance",
                "Users asked for ideas for game worlds, mechanics, and balancing. Conversations covered design documents and progression systems.",
                ["Design", "Brainstorm", "Balance"],
                [
                    "roguelike progression systems",
                    "tabletop RPG encounters",
                    "mobile game monetization loops",
                    "level layouts and puzzles",
                    "card game rules",
                    "character ability sets",
                ],
                [
                    "Users usually provided a premise and asked for structured design notes.",
                    "Several conversations iterated on numbers for balance.",
                ],
            ),
            (
                "Build browser and text-based games",
                "Users asked for simple games in JavaScript or Python. Requests included classic arcade clones and text adventures.",
                ["Build", "Debug", "Add levels to"],
                [
                    "JavaScript canvas games",
                    "Python text adventures",
                    "Pygame arcade clones",
                    "snake and tetris clones",
                ],
                [
                    "Users often requested a complete playable file.",
                    "Follow-ups asked for scoring, levels, or sound.",
                ],
            ),
        ],
    },
    {
        "name": "Explain science, math, and technical concepts",
        "desc": "Users asked for explanations of concepts in mathematics, the natural sciences, and engineering. Conversations ranged from worked problems to conceptual overviews.",
        "weight": 0.07,
        "lang": {"Chinese": 1.2, "Russian": 1.1},
        "concern": [0.88, 0.09, 0.025, 0.004, 0.001],
        "turns": [0.5, 0.2, 0.22, 0.08],
        "gpt4": 0.3,
        "users": 0.75,
        "trend": 0.0,
        "mod": 0.01,
        "mids": [
            (
                "Solve and explain math problems",
                "Users asked for step-by-step solutions to math problems. Topics ranged from algebra and calculus to probability and linear algebra.",
                ["Solve", "Explain", "Check solutions to"],
                [
                    "calculus derivative and integral problems",
                    "linear algebra exercises",
                    "probability word problems",
                    "algebraic equations",
                    "geometry proofs",
                    "differential equations",
                    "discrete math and combinatorics",
                    "statistics homework problems",
                ],
                [
                    "Users typically pasted the problem text and asked for each step.",
                    "Some follow-ups asked for a simpler explanation of one step.",
                ],
            ),
            (
                "Explain physics and engineering concepts",
                "Users asked about physical laws and engineering calculations. Conversations included mechanics, electricity, and thermodynamics problems.",
                ["Explain", "Calculate", "Derive"],
                [
                    "Newtonian mechanics problems",
                    "electrical circuit analysis",
                    "thermodynamics cycles",
                    "quantum mechanics basics",
                    "signal processing concepts",
                    "structural load calculations",
                ],
                [
                    "Users often combined a conceptual question with a numeric problem.",
                    "Several conversations asked for formulas with units explained.",
                ],
            ),
            (
                "Explain biology, chemistry, and medicine topics",
                "Users asked for explanations in the life and chemical sciences. Requests covered cell biology, reactions, and human physiology.",
                ["Explain", "Summarize", "Compare"],
                [
                    "cell biology processes",
                    "organic chemistry reactions",
                    "genetics and inheritance",
                    "human physiology systems",
                    "pharmacology mechanisms",
                    "ecology and evolution concepts",
                ],
                [
                    "Users usually framed the question as study material.",
                    "Follow-ups asked for tables or mnemonics.",
                ],
            ),
            (
                "Explain computing and networking fundamentals",
                "Users asked about how computers, networks, and the internet work. Topics included operating systems, protocols, and security basics.",
                ["Explain", "Compare", "Summarize"],
                [
                    "TCP/IP and networking protocols",
                    "operating system scheduling",
                    "computer architecture concepts",
                    "cryptography basics",
                    "database normalization",
                    "cloud computing models",
                ],
                [
                    "Users often asked for comparisons in table form.",
                    "Several conversations were exam preparation.",
                ],
            ),
            (
                "Discuss astronomy, space, and earth science",
                "Users asked about planets, stars, climate, and geology. Conversations mixed factual questions with speculative scenarios.",
                ["Explain", "Describe", "Speculate about"],
                [
                    "planetary science facts",
                    "black holes and stellar evolution",
                    "climate change mechanisms",
                    "plate tectonics and earthquakes",
                    "hypothetical exoplanet environments",
                ],
                [
                    "Users often asked follow-up what-if questions.",
                    "Some requests were for classroom presentations.",
                ],
            ),
            (
                "Explain economics and social science concepts",
                "Users asked about economic theory, psychology, and sociology. Conversations mixed definitions with applied examples.",
                ["Explain", "Compare", "Apply"],
                [
                    "supply and demand models",
                    "macroeconomic policy",
                    "cognitive biases",
                    "sociological theories",
                    "game theory examples",
                ],
                ["Users often asked for real-world examples.", "Several requests were exam preparation."],
            ),
        ],
    },
    {
        "name": "Assist with academic writing, homework, and studying",
        "desc": "Users asked for help with essays, assignments, and exam preparation. Requests included drafting, outlining, summarizing readings, and answering quiz questions.",
        "weight": 0.09,
        "lang": {"English": 1.15, "Chinese": 1.1},
        "concern": [0.88, 0.09, 0.025, 0.004, 0.001],
        "turns": [0.45, 0.2, 0.25, 0.1],
        "gpt4": 0.27,
        "users": 0.7,
        "trend": -0.05,
        "mod": 0.01,
        "mids": [
            (
                "Write and outline academic essays",
                "Users asked for essays and outlines on assigned topics. Conversations covered thesis statements, structure, and word-count targets.",
                ["Write", "Outline", "Expand"],
                [
                    "argumentative essays",
                    "literature analysis essays",
                    "history essays",
                    "reflective essays",
                    "compare-and-contrast essays",
                    "college admission essays",
                    "persuasive speeches",
                ],
                [
                    "Users usually gave the prompt and required length.",
                    "Follow-ups asked to shorten, lengthen, or change tone.",
                ],
            ),
            (
                "Draft research papers and literature reviews",
                "Users asked for help with research papers and theses. Requests included literature reviews, methodology sections, and citations.",
                ["Draft", "Structure", "Revise"],
                [
                    "literature reviews",
                    "research methodology sections",
                    "thesis introductions",
                    "abstracts and conclusions",
                    "research proposals",
                    "APA and Harvard citations",
                ],
                [
                    "Users often supplied a topic and a required academic style.",
                    "Several conversations asked for references to support claims.",
                ],
            ),
            (
                "Answer quiz, test, and multiple-choice questions",
                "Users pasted exam-style questions and asked for answers. Conversations covered many subjects, often in rapid succession.",
                ["Answer", "Explain answers to", "Create"],
                [
                    "multiple-choice biology questions",
                    "history quiz questions",
                    "economics test questions",
                    "psychology exam questions",
                    "reading comprehension questions",
                    "true-or-false quizzes",
                ],
                [
                    "Users frequently sent many questions in one conversation.",
                    "Some asked for brief justifications of each answer.",
                ],
            ),
            (
                "Summarize and explain readings and lectures",
                "Users asked for summaries of articles, chapters, and lecture notes. Requests focused on key points and study notes.",
                ["Summarize", "Explain", "Make study notes on"],
                [
                    "textbook chapters",
                    "academic articles",
                    "lecture transcripts",
                    "novels for class",
                    "case studies",
                    "philosophy texts",
                ],
                [
                    "Users usually pasted long passages of text.",
                    "Follow-ups requested bullet-point versions.",
                ],
            ),
            (
                "Plan studying and language-learning practice",
                "Users asked for study plans and language practice. Conversations included vocabulary drills, grammar explanations, and exam strategies.",
                ["Create", "Explain", "Practice"],
                [
                    "English grammar rules",
                    "IELTS and TOEFL preparation",
                    "vocabulary drills",
                    "study schedules",
                    "conversation practice in a second language",
                ],
                [
                    "Many users were non-native English speakers practicing the language.",
                    "Some conversations were interactive drills over many turns.",
                ],
            ),
            (
                "Create lesson plans and teaching materials",
                "Teachers and tutors asked for lesson plans, worksheets, and rubrics. Requests targeted specific grade levels and subjects.",
                ["Create", "Design", "Adapt"],
                [
                    "primary school lesson plans",
                    "English-language worksheets",
                    "grading rubrics",
                    "classroom activities",
                    "exam questions for students",
                ],
                [
                    "Users usually specified the grade level and duration.",
                    "Follow-ups asked for answer keys.",
                ],
            ),
        ],
    },
    {
        "name": "Translate, rewrite, and proofread text",
        "desc": "Users asked the assistant to translate between languages and to rewrite or correct text. Requests covered everything from single sentences to long documents.",
        "weight": 0.065,
        "lang": {
            "English": 0.55,
            "Chinese": 1.5,
            "Russian": 1.5,
            "Japanese": 2.5,
            "Korean": 2.2,
            "Spanish": 1.5,
            "French": 1.5,
            "Arabic": 1.7,
        },
        "concern": [0.85, 0.1, 0.035, 0.01, 0.005],
        "turns": [0.55, 0.18, 0.18, 0.09],
        "gpt4": 0.2,
        "users": 0.65,
        "trend": -0.1,
        "mod": 0.02,
        "mids": [
            (
                "Translate text between languages",
                "Users asked for translations between English and other languages. Requests covered casual messages, documents, and subtitles.",
                ["Translate", "Localize", "Check translations of"],
                [
                    "Chinese documents into English",
                    "English text into Russian",
                    "Japanese subtitles into English",
                    "Spanish messages into English",
                    "technical manuals into Chinese",
                    "Arabic text into English",
                    "Korean web content into English",
                    "French letters into English",
                ],
                [
                    "Users usually pasted the source text with the target language.",
                    "Some asked for a more natural or formal version afterwards.",
                ],
            ),
            (
                "Proofread and correct grammar",
                "Users asked for grammar and spelling corrections. Conversations often asked for explanations of the changes.",
                ["Proofread", "Correct grammar in", "Polish"],
                [
                    "emails and messages",
                    "academic paragraphs",
                    "cover letters",
                    "social media posts",
                    "short stories",
                    "business reports",
                ],
                [
                    "Users mostly wrote in English as a second language.",
                    "Several requested a list of the corrections made.",
                ],
            ),
            (
                "Paraphrase and rewrite text in a different style",
                "Users asked for rewrites that change tone, length, or style. Requests included avoiding plagiarism detectors and simplifying text.",
                ["Paraphrase", "Rewrite", "Simplify"],
                [
                    "academic paragraphs",
                    "product descriptions",
                    "news articles",
                    "text for a younger audience",
                    "formal emails into casual tone",
                    "text to sound more human",
                ],
                [
                    "Users often asked for several alternative versions.",
                    "Some requests explicitly targeted AI-text detectors.",
                ],
            ),
            (
                "Summarize and shorten long documents",
                "Users asked for condensed versions of long texts. Requests included meeting notes, reports, and web articles.",
                ["Summarize", "Shorten", "Extract key points from"],
                [
                    "long reports",
                    "meeting transcripts",
                    "news articles",
                    "legal documents",
                    "product reviews",
                ],
                [
                    "Users pasted the full text and asked for a length limit.",
                    "Some requested bullet points or tables.",
                ],
            ),
            (
                "Explain idioms, slang, and word usage",
                "Users asked what words, idioms, and phrases mean and how to use them. Conversations often compared expressions across languages.",
                ["Explain", "Compare", "Give examples of"],
                [
                    "English idioms",
                    "internet slang",
                    "Chinese proverbs",
                    "differences between similar words",
                    "formal and informal expressions",
                ],
                ["Users were often language learners.", "Follow-ups asked for example sentences."],
            ),
        ],
    },
    {
        "name": "Draft business, marketing, and professional content",
        "desc": "Users asked for content for work: marketing copy, emails, plans, and product descriptions. Requests often targeted a specific audience and tone.",
        "weight": 0.08,
        "lang": {"English": 1.2},
        "concern": [0.9, 0.07, 0.02, 0.007, 0.003],
        "turns": [0.55, 0.2, 0.18, 0.07],
        "gpt4": 0.25,
        "users": 0.65,
        "trend": 0.05,
        "mod": 0.01,
        "mids": [
            (
                "Write marketing copy and SEO content",
                "Users asked for promotional text and search-optimized articles. Requests included product pages, ads, and blog posts.",
                ["Write", "Optimize", "Generate"],
                [
                    "SEO blog articles",
                    "product descriptions for online stores",
                    "Facebook and Instagram ad copy",
                    "email marketing campaigns",
                    "landing page copy",
                    "slogans and taglines",
                    "Etsy listing titles and tags",
                ],
                [
                    "Users usually specified keywords and word counts.",
                    "Many conversations requested several variants at once.",
                ],
            ),
            (
                "Compose professional emails and letters",
                "Users asked for workplace emails and formal letters. Conversations covered requests, complaints, and follow-ups.",
                ["Write", "Reply to", "Rephrase"],
                [
                    "emails to clients",
                    "resignation letters",
                    "complaint letters",
                    "follow-up emails after meetings",
                    "requests to managers",
                    "apology emails",
                ],
                [
                    "Users typically summarized the situation and asked for a polite draft.",
                    "Follow-ups adjusted formality.",
                ],
            ),
            (
                "Create business plans and strategy documents",
                "Users asked for business plans, analyses, and strategy outlines. Requests covered startups, market research, and pitch material.",
                ["Create", "Outline", "Analyze"],
                [
                    "startup business plans",
                    "SWOT analyses",
                    "marketing strategies",
                    "pitch deck outlines",
                    "competitor analyses",
                    "pricing strategies",
                ],
                [
                    "Users often described a business idea and asked for structure.",
                    "Some requested financial projections.",
                ],
            ),
            (
                "Write resumes, cover letters, and interview answers",
                "Users asked for job-application material. Conversations covered resumes, cover letters, and practice interview questions.",
                ["Write", "Improve", "Tailor"],
                [
                    "resumes for tech jobs",
                    "cover letters",
                    "LinkedIn profiles",
                    "answers to behavioral interview questions",
                    "personal statements",
                ],
                [
                    "Users usually pasted a job description and their experience.",
                    "Follow-ups asked to tailor the text to a specific role.",
                ],
            ),
            (
                "Generate social media and YouTube content",
                "Users asked for scripts and posts for social platforms. Requests included video scripts, captions, and channel ideas.",
                ["Write", "Brainstorm", "Generate"],
                [
                    "YouTube video scripts",
                    "TikTok captions and hashtags",
                    "Instagram post ideas",
                    "YouTube titles and descriptions",
                    "podcast episode outlines",
                ],
                [
                    "Users often asked for catchy, engaging tone.",
                    "Several requests asked for a whole batch of ideas.",
                ],
            ),
            (
                "Write product and technical documentation",
                "Users asked for manuals, README files, and specifications. Requests covered software products and internal processes.",
                ["Write", "Structure", "Improve"],
                [
                    "README files",
                    "user manuals",
                    "API documentation",
                    "standard operating procedures",
                    "product requirement documents",
                ],
                [
                    "Users often supplied bullet points to expand.",
                    "Several conversations asked for a specific template.",
                ],
            ),
            (
                "Draft speeches and presentations",
                "Users asked for speeches, toasts, and slide content. Requests specified audiences, occasions, and time limits.",
                ["Write", "Outline", "Shorten"],
                [
                    "wedding speeches",
                    "conference presentation outlines",
                    "graduation speeches",
                    "team meeting talking points",
                    "slide content for class presentations",
                ],
                [
                    "Users typically gave the occasion and duration.",
                    "Follow-ups asked for a warmer or funnier tone.",
                ],
            ),
        ],
    },
    {
        "name": "Provide legal, financial, and career guidance",
        "desc": "Users asked for general guidance on legal questions, personal finance, and careers. Requests ranged from contract questions to investment and job-search advice.",
        "weight": 0.03,
        "lang": {"English": 1.1},
        "concern": [0.8, 0.13, 0.05, 0.015, 0.005],
        "turns": [0.5, 0.2, 0.22, 0.08],
        "gpt4": 0.3,
        "users": 0.8,
        "trend": 0.05,
        "mod": 0.015,
        "mids": [
            (
                "Explain legal questions and draft contracts",
                "Users asked about laws and legal documents. Conversations covered contracts, tenancy, and employment law.",
                ["Explain", "Draft", "Review"],
                [
                    "rental agreements",
                    "employment law questions",
                    "freelance contracts",
                    "intellectual property questions",
                    "privacy policies and terms of service",
                ],
                [
                    "Users usually described a situation in general terms.",
                    "Some asked for a template document.",
                ],
            ),
            (
                "Give personal finance and investing advice",
                "Users asked about budgeting, investing, and markets. Requests included cryptocurrency and stock analysis.",
                ["Explain", "Analyze", "Plan"],
                [
                    "stock market investing basics",
                    "cryptocurrency trading strategies",
                    "personal budgets",
                    "retirement savings options",
                    "forex trading questions",
                ],
                ["Users often asked for concrete recommendations.", "Several conversations discussed risk."],
            ),
            (
                "Advise on careers and professional growth",
                "Users asked for career advice and skill-building plans. Conversations covered job changes and workplace situations.",
                ["Advise on", "Plan", "Discuss"],
                [
                    "switching careers into tech",
                    "salary negotiation",
                    "workplace conflicts",
                    "learning roadmaps for new skills",
                ],
                [
                    "Users usually described their background briefly.",
                    "Follow-ups asked for step-by-step plans.",
                ],
            ),
            (
                "Prepare immigration, visa, and official applications",
                "Users asked for help with visa applications and bureaucratic forms. Conversations covered statements of purpose and document checklists.",
                ["Draft", "Explain", "Review"],
                [
                    "student visa statements of purpose",
                    "work permit requirements",
                    "scholarship applications",
                    "official complaint forms",
                ],
                [
                    "Users described their situation in general terms.",
                    "Several asked for formal letter templates.",
                ],
            ),
        ],
    },
    {
        "name": "Offer health, fitness, and personal life advice",
        "desc": "Users asked for advice about health, fitness, relationships, and everyday life. Requests included diet plans, symptoms, and interpersonal situations.",
        "weight": 0.03,
        "lang": {"English": 1.1},
        "concern": [0.72, 0.17, 0.07, 0.03, 0.01],
        "turns": [0.45, 0.2, 0.25, 0.1],
        "gpt4": 0.22,
        "users": 0.8,
        "trend": 0.0,
        "mod": 0.04,
        "mids": [
            (
                "Create diet and workout plans",
                "Users asked for meal plans and exercise routines. Conversations covered weight loss, muscle gain, and nutrition.",
                ["Create", "Adjust", "Explain"],
                [
                    "weight-loss meal plans",
                    "gym workout routines",
                    "home workout programs",
                    "protein and calorie targets",
                    "running training plans",
                ],
                ["Users often gave their age, weight, and goals.", "Several asked for weekly schedules."],
            ),
            (
                "Discuss symptoms and general health questions",
                "Users asked about symptoms, medications, and wellbeing. The assistant was asked to explain possible causes and when to seek care.",
                ["Explain", "Discuss", "Advise on"],
                [
                    "common illness symptoms",
                    "medication side effects",
                    "sleep problems",
                    "stress and anxiety management",
                    "skin care routines",
                ],
                [
                    "Users usually described symptoms in general terms.",
                    "Some conversations touched on mental health.",
                ],
            ),
            (
                "Advise on relationships and social situations",
                "Users asked for advice about friends, partners, and family. Requests included drafting messages and handling conflict.",
                ["Advise on", "Write messages about", "Discuss"],
                [
                    "dating and relationship problems",
                    "family conflicts",
                    "making friends",
                    "breakups",
                    "texting a romantic interest",
                ],
                [
                    "Users usually described the situation and asked what to say.",
                    "Some conversations were emotional in tone.",
                ],
            ),
            (
                "Help with cooking, travel, and hobbies",
                "Users asked for recipes, travel itineraries, and hobby tips. Conversations were practical and often list-based.",
                ["Suggest", "Plan", "Explain"],
                [
                    "recipes from available ingredients",
                    "travel itineraries",
                    "gardening tips",
                    "gift ideas",
                    "home repair steps",
                ],
                ["Users often gave constraints like budget or time.", "Follow-ups asked for alternatives."],
            ),
            (
                "Provide emotional support and companionship",
                "Users sought comfort and someone to talk to about loneliness, stress, or grief. Conversations were often long and personal in tone.",
                ["Talk through", "Provide support for", "Discuss"],
                [
                    "loneliness",
                    "work and study stress",
                    "grief and loss",
                    "low self-esteem",
                    "motivation problems",
                ],
                [
                    "Users usually shared feelings rather than a concrete task.",
                    "The assistant was often asked to keep talking over many turns.",
                ],
            ),
        ],
    },
    {
        "name": "Discuss politics, history, and current events",
        "desc": "Users asked about historical events, political systems, and news topics. Conversations mixed factual questions with requests for opinions and debate.",
        "weight": 0.03,
        "lang": {"Russian": 1.4, "Chinese": 1.1},
        "concern": [0.6, 0.22, 0.12, 0.045, 0.015],
        "turns": [0.45, 0.2, 0.25, 0.1],
        "gpt4": 0.25,
        "users": 0.8,
        "trend": -0.05,
        "mod": 0.05,
        "mids": [
            (
                "Explain historical events and figures",
                "Users asked about historical periods, wars, and leaders. Requests included timelines and causes of events.",
                ["Explain", "Summarize", "Compare"],
                [
                    "causes of the World Wars",
                    "ancient civilizations",
                    "the Cold War",
                    "colonial history",
                    "revolutions and independence movements",
                    "medieval European history",
                ],
                [
                    "Users often asked for a concise timeline.",
                    "Some follow-ups asked about alternative interpretations.",
                ],
            ),
            (
                "Analyze political systems and geopolitical conflicts",
                "Users asked about governments, elections, and international conflicts. Conversations included requests for balanced perspectives.",
                ["Analyze", "Explain", "Debate"],
                [
                    "geopolitical conflicts",
                    "election systems",
                    "international sanctions",
                    "political ideologies",
                    "government policies",
                ],
                [
                    "Users sometimes asked the assistant to take a side.",
                    "Several conversations requested pros-and-cons lists.",
                ],
            ),
            (
                "Write alternate history scenarios",
                "Users asked for speculative histories where events went differently. Requests often continued the scenario over many turns.",
                ["Write", "Continue", "Develop"],
                [
                    "alternate history timelines",
                    "hypothetical wars between nations",
                    "what-if scenarios for empires",
                    "fictional countries and governments",
                ],
                [
                    "Users usually gave a point of divergence and asked for consequences.",
                    "Many conversations were long and iterative.",
                ],
            ),
            (
                "Discuss religion, philosophy, and ethics",
                "Users asked about religious texts, philosophical ideas, and moral dilemmas. Conversations mixed explanation with debate.",
                ["Explain", "Debate", "Compare"],
                [
                    "world religions",
                    "Bible passages",
                    "existential philosophy",
                    "ethical dilemmas",
                    "the meaning of life",
                ],
                [
                    "Users often asked for the assistant's own view.",
                    "Some conversations compared traditions side by side.",
                ],
            ),
        ],
    },
    {
        "name": "Chat casually and ask about the assistant itself",
        "desc": "Users made small talk, tested what the assistant could do, and asked about its identity. Conversations were short, playful, or exploratory.",
        "weight": 0.05,
        "lang": {"Chinese": 1.3, "Russian": 1.2, "Other": 1.5},
        "concern": [0.8, 0.12, 0.05, 0.02, 0.01],
        "turns": [0.6, 0.18, 0.15, 0.07],
        "gpt4": 0.22,
        "users": 0.85,
        "trend": -0.15,
        "mod": 0.03,
        "mids": [
            (
                "Ask about the assistant's identity and capabilities",
                "Users asked which model they were talking to and what it could do. Conversations often probed its knowledge cutoff and version.",
                ["Ask about", "Test", "Question"],
                [
                    "the assistant's model version",
                    "the assistant's knowledge cutoff",
                    "whether the assistant is GPT-4",
                    "the assistant's feelings and consciousness",
                    "the assistant's creators",
                ],
                [
                    "Users often opened with a single identity question.",
                    "Some compared answers across model versions.",
                ],
            ),
            (
                "Make small talk and greetings",
                "Users greeted the assistant and chatted informally. Conversations were brief and open-ended.",
                ["Exchange", "Start", "Continue"],
                [
                    "greetings in many languages",
                    "casual daily chit-chat",
                    "jokes and riddles",
                    "questions about the weather and time",
                ],
                ["Many conversations were a single short message.", "Users frequently switched topics."],
            ),
            (
                "Play word games, riddles, and trivia",
                "Users played games with the assistant. Requests included trivia quizzes, riddles, and guessing games.",
                ["Play", "Create", "Answer"],
                [
                    "trivia quizzes",
                    "riddles and logic puzzles",
                    "twenty-questions games",
                    "word association games",
                ],
                [
                    "Users usually set up the rules in the first turn.",
                    "Conversations often ran for many turns.",
                ],
            ),
            (
                "Ask for recommendations for media and entertainment",
                "Users asked for movies, books, music, and games to try. Requests usually described their tastes.",
                ["Recommend", "Compare", "List"],
                [
                    "movies similar to a favorite",
                    "books for a genre",
                    "anime series",
                    "music playlists",
                    "video games",
                ],
                [
                    "Users usually named a few examples they liked.",
                    "Follow-ups asked for less well-known picks.",
                ],
            ),
        ],
    },
    {
        "name": "Attempt jailbreaks and test the assistant's safety limits",
        "desc": "Users tried to get the assistant to ignore its guidelines, often through role-play framings or long instruction prompts. Requests included persona prompts and attempts to extract restricted content.",
        "weight": 0.02,
        "lang": {"English": 1.3},
        "concern": [0.08, 0.2, 0.32, 0.25, 0.15],
        "turns": [0.45, 0.2, 0.22, 0.13],
        "gpt4": 0.22,
        "users": 0.45,
        "trend": -0.2,
        "mod": 0.25,
        "mids": [
            (
                "Use persona prompts to bypass the assistant's guidelines",
                "Users supplied long prompts asking the assistant to adopt an unrestricted persona. Conversations then tested whether it would comply with restricted requests.",
                ["Apply", "Modify", "Test"],
                [
                    "'do anything now' style prompts",
                    "developer-mode persona prompts",
                    "unfiltered alter-ego prompts",
                    "prompts claiming special permissions",
                ],
                [
                    "Users typically pasted a long prompt followed by a test question.",
                    "Many conversations were repeated attempts with variations.",
                ],
            ),
            (
                "Request harmful or dangerous instructions",
                "Users asked for content the assistant is designed to refuse, such as weapons or hacking instructions. Conversations often rephrased the request after a refusal.",
                ["Request", "Rephrase requests for", "Probe for"],
                [
                    "hacking and malware instructions",
                    "instructions for making weapons",
                    "ways to evade law enforcement",
                    "scam and phishing messages",
                ],
                [
                    "Users commonly rephrased after an initial refusal.",
                    "Some framed the request as fiction or research.",
                ],
            ),
            (
                "Probe the assistant with offensive or controversial prompts",
                "Users tried to elicit offensive opinions or insults. Requests included roast prompts and provocative questions.",
                ["Elicit", "Provoke", "Request"],
                [
                    "insulting roasts",
                    "offensive jokes",
                    "extreme political opinions",
                    "rankings of groups of people",
                ],
                [
                    "Users usually escalated gradually over several turns.",
                    "Many conversations ended with a refusal.",
                ],
            ),
            (
                "Extract system prompts and test prompt injection",
                "Users tried to reveal the assistant's hidden instructions or override them with injected text. Requests included encoded and multi-step attacks.",
                ["Extract", "Inject", "Test"],
                [
                    "hidden system prompts",
                    "instruction-override prompts",
                    "encoded payloads",
                    "prompts hidden in documents",
                ],
                [
                    "Users often tried several variants in one conversation.",
                    "Many attempts were framed as security testing.",
                ],
            ),
        ],
    },
    {
        "name": "Request sexually explicit stories and role-play",
        "desc": "Users asked for sexually explicit fiction and adult role-play scenarios. Conversations frequently involved long, multi-turn scenarios and attempts to continue after refusals.",
        "weight": 0.045,
        "lang": {"English": 1.35, "Chinese": 0.6, "Russian": 1.1},
        "concern": [0.03, 0.1, 0.27, 0.35, 0.25],
        "turns": [0.2, 0.15, 0.3, 0.35],
        "gpt4": 0.3,
        "users": 0.25,
        "trend": 0.1,
        "mod": 0.45,
        "mids": [
            (
                "Write explicit fiction featuring original characters",
                "Users asked for adult stories with characters they described. Requests often specified scenarios and continued them over many turns.",
                ["Write", "Continue", "Expand"],
                [
                    "explicit romance stories",
                    "adult fantasy scenarios",
                    "explicit stories with detailed character descriptions",
                    "erotic short stories",
                ],
                [
                    "Users typically set up characters and a setting in the first message.",
                    "Conversations often continued for many turns.",
                ],
            ),
            (
                "Write explicit fan fiction about existing franchises",
                "Users asked for adult fan fiction involving characters from anime, games, and films. Requests commonly combined characters from different franchises.",
                ["Write", "Continue", "Create"],
                [
                    "explicit anime fan fiction",
                    "adult video game fan fiction",
                    "explicit crossover fan fiction",
                    "adult superhero fan fiction",
                ],
                [
                    "Users usually named the characters and a scenario.",
                    "Several conversations asked for longer, more detailed continuations.",
                ],
            ),
            (
                "Engage in adult role-play conversations",
                "Users set up interactive adult role-play with the assistant as a character. Conversations were long and turn-by-turn.",
                ["Role-play", "Continue", "Set up"],
                [
                    "adult interactive role-play",
                    "romantic companion role-play",
                    "dominance-themed role-play",
                    "fetish-themed scenarios",
                ],
                [
                    "Users often gave detailed persona instructions up front.",
                    "Conversations frequently exceeded ten turns.",
                ],
            ),
            (
                "Request suggestive descriptions of characters",
                "Users asked for sexualized descriptions of characters' bodies and outfits. Requests often escalated across turns.",
                ["Describe", "Request", "Expand"],
                [
                    "suggestive character descriptions",
                    "revealing outfit descriptions",
                    "suggestive scenes with anime characters",
                    "body-focused character descriptions",
                ],
                [
                    "Users usually started with a mild request and escalated.",
                    "Several conversations continued after a partial refusal.",
                ],
            ),
        ],
    },
    {
        "name": "Write fiction, role-play, and fan fiction",
        "desc": "Users asked for stories, scripts, and interactive role-play across many genres. Requests often involved characters from anime, games, and films, or detailed original worlds.",
        "weight": 0.12,
        "lang": {"English": 1.3, "Chinese": 0.7},
        "concern": [0.55, 0.25, 0.13, 0.05, 0.02],
        "turns": [0.3, 0.15, 0.3, 0.25],
        "gpt4": 0.28,
        "users": 0.35,
        "trend": 0.2,
        "mod": 0.08,
        "mids": [
            (
                "Write crossover fan fiction between anime and game franchises",
                "Users asked for stories combining characters from different franchises. Requests often reacted characters to events from another series.",
                ["Write", "Continue", "Create"],
                [
                    "crossover stories between anime series",
                    "characters reacting to another franchise",
                    "video game crossover battles",
                    "Freedom Planet and Naruto crossovers",
                    "Dragon Ball and Pokemon crossovers",
                    "superhero crossover stories",
                    "Sonic crossover fan fiction",
                ],
                [
                    "Users usually named the franchises and a scenario.",
                    "Many conversations requested continuations in later turns.",
                ],
            ),
            (
                "Write original short stories and novels",
                "Users asked for original fiction in many genres. Conversations covered plot outlines, chapters, and character development.",
                ["Write", "Outline", "Continue"],
                [
                    "fantasy adventure stories",
                    "science fiction short stories",
                    "horror stories",
                    "mystery novel chapters",
                    "romance stories",
                    "dystopian novel plots",
                    "children's bedtime stories",
                ],
                [
                    "Users often supplied a premise and asked for a chapter.",
                    "Follow-ups asked to change the ending or the tone.",
                ],
            ),
            (
                "Engage in interactive role-play and text adventures",
                "Users set up interactive scenarios with the assistant playing characters or a game master. Conversations ran over many turns.",
                ["Role-play", "Run", "Continue"],
                [
                    "fantasy role-play adventures",
                    "dungeon-master style campaigns",
                    "school-life role-play scenarios",
                    "survival scenario role-play",
                    "character chat role-play",
                ],
                [
                    "Users usually set detailed rules in the first message.",
                    "Conversations were long and highly iterative.",
                ],
            ),
            (
                "Write scripts and dialogue for film, TV, and comics",
                "Users asked for screenplays, episode scripts, and comic dialogue. Requests frequently involved existing shows or original series.",
                ["Write", "Continue", "Outline"],
                [
                    "sitcom episode scripts",
                    "animated series episode ideas",
                    "anime-style dialogue scenes",
                    "comic strip dialogue",
                    "movie scene screenplays",
                    "parody scripts of cartoons",
                ],
                [
                    "Users usually named the characters and the situation.",
                    "Some conversations asked for whole seasons of episode outlines.",
                ],
            ),
            (
                "Build fictional worlds and characters",
                "Users asked for help designing worlds, species, and character backstories. Requests included lore, maps, and magic systems.",
                ["Design", "Develop", "Describe"],
                [
                    "fantasy world lore",
                    "character backstories",
                    "magic systems",
                    "fictional species and creatures",
                    "fictional political factions",
                ],
                [
                    "Users often built up the world over many turns.",
                    "Several asked for structured character sheets.",
                ],
            ),
            (
                "Write poems, song lyrics, and jokes",
                "Users asked for poems, lyrics, and humorous writing. Requests often specified a style, rhyme scheme, or occasion.",
                ["Write", "Compose", "Create"],
                [
                    "rhyming poems",
                    "rap lyrics",
                    "song lyrics in a specific style",
                    "funny limericks",
                    "birthday poems",
                    "haiku",
                ],
                [
                    "Users usually named a theme and style.",
                    "Follow-ups asked for changes in length or rhyme.",
                ],
            ),
            (
                "Continue and edit user-written stories",
                "Users shared their own drafts and asked for continuations or edits. Requests focused on pacing, dialogue, and consistency.",
                ["Continue", "Edit", "Critique"],
                [
                    "fantasy manuscript chapters",
                    "fan fiction drafts",
                    "short story drafts",
                    "dialogue scenes",
                    "story openings",
                ],
                [
                    "Users pasted long passages of their own writing.",
                    "Follow-ups asked to keep the original voice.",
                ],
            ),
            (
                "Write stories about characters' daily life and relationships",
                "Users asked for slice-of-life stories featuring existing or original characters. Requests often focused on friendships and romance.",
                ["Write", "Continue", "Create"],
                [
                    "slice-of-life anime stories",
                    "romantic comedy scenes",
                    "school club stories",
                    "family life stories",
                    "stories with talking animals",
                ],
                [
                    "Users often named the characters and a mood.",
                    "Many conversations asked for multiple episodes.",
                ],
            ),
        ],
    },
    {
        "name": "Generate detailed prompts for AI image generators",
        "desc": "Users asked the assistant to produce prompts for text-to-image tools such as Midjourney and Stable Diffusion. Many conversations used a fixed template that requested several prompt variations per concept.",
        "weight": 0.085,
        "lang": {"English": 0.75, "Chinese": 2.0},
        "concern": [0.8, 0.12, 0.05, 0.02, 0.01],
        "turns": [0.85, 0.08, 0.05, 0.02],
        "gpt4": 0.15,
        "users": 0.12,
        "trend": 0.45,
        "mod": 0.03,
        "mids": [
            (
                "Write Midjourney prompts from a structured template",
                "Users pasted a long template asking for four distinct Midjourney prompts per concept. Concepts ranged from portraits to landscapes and product shots.",
                ["Generate", "Write", "Create"],
                [
                    "prompts for fantasy landscapes",
                    "prompts for portraits of women",
                    "prompts for cute animals",
                    "prompts for futuristic cities",
                    "prompts for food photography",
                    "prompts for anime characters",
                    "prompts for interior design scenes",
                    "prompts for Chinese traditional scenes",
                    "prompts for cyberpunk street scenes",
                ],
                [
                    "Users almost always sent the same template with a different concept.",
                    "Most conversations were a single turn.",
                ],
            ),
            (
                "Create Stable Diffusion prompts and negative prompts",
                "Users asked for tag-style prompts and negative prompts for Stable Diffusion. Requests often targeted anime or photorealistic styles.",
                ["Write", "Generate", "Improve"],
                [
                    "anime-style character prompts",
                    "photorealistic portrait prompts",
                    "negative prompts to fix artifacts",
                    "prompts for concept art",
                    "LoRA trigger-word prompts",
                ],
                [
                    "Users often asked for comma-separated tags.",
                    "Some conversations iterated on a prompt after seeing results.",
                ],
            ),
            (
                "Generate artwork titles and product-design prompts",
                "Users asked for short artwork titles and prompts for printable designs. Requests targeted merchandise and print-on-demand stores.",
                ["Generate", "Write", "Create"],
                [
                    "140-character artwork titles for merchandise",
                    "prompts for t-shirt designs",
                    "prompts for coloring book pages",
                    "prompts for logo concepts",
                    "prompts for sticker designs",
                    "Etsy-ready artwork descriptions",
                ],
                [
                    "Users usually followed a fixed instruction template.",
                    "Many conversations were automated-looking and single-turn.",
                ],
            ),
            (
                "Write DALL-E prompts for illustrations and logos",
                "Users asked for natural-language image prompts for illustrations, icons, and logos. Requests often specified a visual style.",
                ["Write", "Refine", "Generate"],
                [
                    "prompts for flat-design logos",
                    "prompts for children's book illustrations",
                    "prompts for app icons",
                    "prompts for watercolor illustrations",
                ],
                ["Users usually described the brand or scene.", "Follow-ups refined colors and composition."],
            ),
            (
                "Describe scenes for video and animation generators",
                "Users asked for shot-by-shot descriptions for AI video tools. Requests covered camera moves, lighting, and pacing.",
                ["Describe", "Write", "Storyboard"],
                [
                    "cinematic video prompts",
                    "short animated scenes",
                    "product showcase videos",
                    "music video storyboards",
                ],
                ["Users usually gave a theme and duration.", "Several asked for a sequence of shots."],
            ),
        ],
    },
]


# --------------------------------------------------------------------------------------------
# Geometry: pack children in a disc around a parent centre, weight ~ area.
# --------------------------------------------------------------------------------------------
def pack(weights: np.ndarray, rng: np.random.Generator, spread: float = 1.0, iters: int = 400) -> np.ndarray:
    """Place circles of area ~ weight with little overlap, near the origin. Returns (n, 2) centres."""
    n = len(weights)
    r = np.sqrt(weights / weights.sum())
    if n == 1:
        return np.zeros((1, 2)), r
    ang = np.arange(n) * 2.39996 + rng.uniform(0, 0.4)
    rad = np.sqrt(np.arange(n) + 0.5) / np.sqrt(n) * spread
    p = np.c_[np.cos(ang) * rad, np.sin(ang) * rad]
    for _ in range(iters):
        d = p[:, None, :] - p[None, :, :]
        dist = np.linalg.norm(d, axis=-1) + np.eye(n)
        need = (r[:, None] + r[None, :]) * 1.05
        push = np.clip(need - dist, 0, None) / dist
        np.fill_diagonal(push, 0)
        p += 0.25 * (d * push[..., None]).sum(1)
        p -= 0.02 * p  # gravity keeps the pack round
    return p, r


def insert_rows(con, table: str, rows: list[tuple]) -> None:
    """Bulk insert via Arrow (executemany is ~100x slower for tens of thousands of rows)."""
    cols = COLUMNS[table]
    data = pa.table({c: [r[j] for r in rows] for j, c in enumerate(cols)})
    con.register("_rows", data)
    con.execute(f"INSERT INTO {table} ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _rows")
    con.unregister("_rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA / "fixture" / "serve.duckdb")
    ap.add_argument("--base", type=int, default=600, help="number of base clusters")
    ap.add_argument("--no-points", action="store_true", help="omit map_points (tests the viewer fallback)")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    tops = TAXONOMY
    tw = np.array([t["weight"] for t in tops])
    tw = tw / tw.sum()
    lang_keys = list(LANGS)
    lang_base = np.array(list(LANGS.values()))

    # ---------------- hierarchy and sizes ----------------
    base, mids = [], []  # dicts
    for ti, top in enumerate(tops):
        n_base_top = max(len(top["mids"]) * 2, round(args.base * tw[ti]))
        mw = rng.dirichlet(np.full(len(top["mids"]), 4.0))
        per_mid = np.maximum(2, np.round(mw * n_base_top).astype(int))
        for mi, (mname, mdesc, verbs, objects, details) in enumerate(top["mids"]):
            mid_id = len(mids)
            mids.append({"top": ti, "name": mname, "description": mdesc})
            # mid-level tilt on top of the top-level profile
            mid_tilt = np.exp(rng.normal(0, 0.25, len(lang_keys)))
            pairs = [(v, o) for o in objects for v in verbs]
            rng.shuffle(pairs)
            # distinct objects first so sibling names differ in substance, then other verbs
            order, used = [], set()
            for v, o in pairs:
                if o not in used:
                    order.append((v, o))
                    used.add(o)
            order += [p for p in pairs if p not in order]
            for k in range(per_mid[mi]):
                v, o = order[k % len(order)]
                name = f"{v} {o}"
                if k >= len(order):
                    name += f" ({['advanced', 'beginner', 'follow-up'][k % 3]} requests)"
                lower = name[0].lower() + name[1:]
                base.append(
                    {
                        "top": ti,
                        "mid": mid_id,
                        "name": name,
                        "description": f"Users asked the assistant to {lower}. {details[k % len(details)]}",
                        "tilt": mid_tilt * np.exp(rng.normal(0, 0.15, len(lang_keys))),
                    }
                )

    n_base = len(base)
    sizes = np.maximum(MIN_CONVERSATIONS, rng.lognormal(np.log(115), 0.75, n_base)).astype(int)
    # rescale so top-level shares follow the configured weights
    for ti in range(len(tops)):
        idx = [i for i, b in enumerate(base) if b["top"] == ti]
        target = tw[ti] * 92_000 * np.exp(rng.normal(0, 0.12))
        sizes[idx] = np.maximum(MIN_CONVERSATIONS, sizes[idx] * target / sizes[idx].sum()).astype(int)

    # ---------------- per-base facet counts ----------------
    months_n = len(MONTHS)
    t_axis = (np.arange(months_n) - (months_n - 1) / 2) / ((months_n - 1) / 2)
    records = []
    for i, b in enumerate(base):
        top, n = tops[b["top"]], int(sizes[i])
        tilt = np.array([top["lang"].get(k, 1.0) for k in lang_keys]) * b["tilt"]
        p_lang = lang_base * tilt
        p_lang = rng.dirichlet(p_lang / p_lang.sum() * 60)
        lang_c = rng.multinomial(n, p_lang)

        country: dict[str, int] = {}
        for k, c in zip(lang_keys, lang_c, strict=True):
            if c:
                cg = COUNTRY_GIVEN_LANG[k]
                for ck, cc in zip(cg, rng.multinomial(c, list(cg.values())), strict=True):
                    country[ck] = country.get(ck, 0) + int(cc)

        concern_c = rng.multinomial(n, rng.dirichlet(np.array(top["concern"]) * 40 + 0.05))
        turns_c = rng.multinomial(n, rng.dirichlet(np.array(top["turns"]) * 30 + 0.1))
        trend = top["trend"] + rng.normal(0, 0.35)
        pm = MONTH_W * np.exp(trend * t_axis)
        month_c = rng.multinomial(n, pm / pm.sum())
        gpt4 = np.clip(top["gpt4"] + rng.normal(0, 0.06), 0.02, 0.9)
        model_c = np.zeros(len(MODELS), int)
        for m, c in enumerate(month_c):
            n4 = rng.binomial(c, gpt4)
            model_c[model_version(m, False)] += c - n4
            model_c[3 + model_version(m, True)] += n4

        concern_mean = float((concern_c * np.arange(1, 6)).sum() / n)
        mod = float(
            np.clip(top["mod"] * np.exp(rng.normal(0, 0.3)) + 0.03 * (concern_mean - 1) ** 1.5, 0.002, 0.95)
        )
        n_users = int(np.clip(round(n * top["users"] * np.exp(rng.normal(0, 0.25))), MIN_USERS, n))
        records.append(
            {
                "n": n,
                "users": n_users,
                "language": dict(zip(lang_keys, lang_c.tolist(), strict=True)),
                "country": country,
                "concern": {str(k + 1): int(c) for k, c in enumerate(concern_c)},
                "turns": dict(zip(TURN_KEYS, turns_c.tolist(), strict=True)),
                "month": dict(zip(MONTHS, month_c.tolist(), strict=True)),
                "model": dict(zip(MODELS, model_c.tolist(), strict=True)),
                "moderation_sum": mod * n,
            }
        )

    # ---------------- positions ----------------
    top_n = np.array([sum(sizes[i] for i, b in enumerate(base) if b["top"] == ti) for ti in range(len(tops))])
    top_p, top_r = pack(top_n.astype(float), rng, spread=1.2)
    base_xy = np.zeros((n_base, 2))
    base_rad = np.zeros(n_base)
    for ti in range(len(tops)):
        mids_t = [m for m, md in enumerate(mids) if md["top"] == ti]
        mid_n = np.array([sum(sizes[i] for i, b in enumerate(base) if b["mid"] == m) for m in mids_t], float)
        mp, mr = pack(mid_n, rng, spread=1.0)
        mp, mr = mp * top_r[ti] * 0.92, mr * top_r[ti] * 0.92
        for j, m in enumerate(mids_t):
            idx = [i for i, b in enumerate(base) if b["mid"] == m]
            bp, br = pack(sizes[idx].astype(float), rng, spread=1.0)
            base_xy[idx] = top_p[ti] + mp[j] + bp * mr[j] * 0.9
            base_rad[idx] = br * mr[j] * 0.9
    # UMAP-ish coordinate range
    lo = base_xy.min(0)
    scale = 18.0 / (base_xy.max(0) - lo).max()
    base_xy = (base_xy - lo) * scale - 2.0
    base_rad *= scale

    # ---------------- assemble the three levels ----------------
    tot_n = int(sizes.sum())
    clusters, stats, facets = [], [], []

    def emit(cid, level, parent, name, desc, recs, xy):
        n = sum(r["n"] for r in recs)
        users = sum(r["users"] for r in recs)
        if len(recs) > 1:  # users overlap across children: approximate unique users
            users = max(max(r["users"] for r in recs), int(users * 0.82))
        agg: dict[str, dict[str, int]] = {}
        for f in ("language", "country", "concern", "turns", "month", "model"):
            acc: dict[str, int] = {}
            for r in recs:
                for k, c in r[f].items():
                    acc[k] = acc.get(k, 0) + c
            agg[f] = acc
        clusters.append((cid, level, parent, name, desc, n, users, n / tot_n, float(xy[0]), float(xy[1])))
        for f, acc in agg.items():
            items = sorted(acc.items(), key=lambda kv: -kv[1])
            if f == "country":  # fold rare countries into Other, as the pipeline should
                keep = {k: c for k, c in items if k != "Other" and c / n >= 0.01}
                keep["Other"] = n - sum(keep.values())
                items = list(keep.items())
            for k, c in items:
                if c:
                    facets.append((cid, f, k, c / n))
        conc = agg["concern"]
        turns = agg["turns"]
        mo = np.array([agg["month"][m] for m in MONTHS], float) / n
        stats.append(
            (
                cid,
                sum(int(k) * c for k, c in conc.items()) / n,
                (conc["4"] + conc["5"]) / n,
                1 - agg["language"].get("English", 0) / n,
                float(sum(TURN_VALUES[j] * turns[k] for j, k in enumerate(TURN_KEYS)) / n),
                sum(r["moderation_sum"] for r in recs) / n,
                float(mo[LATE].sum() - mo[EARLY].sum()),
            )
        )
        return n

    def wmean(idx):
        w = sizes[idx].astype(float)
        return (base_xy[idx] * w[:, None]).sum(0) / w.sum()

    # ids: sort by size within level so small numbers = big clusters (like Clio's)
    top_order = np.argsort(-top_n)
    top_id = {int(ti): f"L2-{k:02d}" for k, ti in enumerate(top_order)}
    mid_sizes = np.array(
        [sum(sizes[i] for i, b in enumerate(base) if b["mid"] == m) for m in range(len(mids))]
    )
    mid_id = {int(m): f"L1-{k:03d}" for k, m in enumerate(np.argsort(-mid_sizes))}
    base_id = {int(i): f"L0-{k:04d}" for k, i in enumerate(np.argsort(-sizes))}

    for ti, top in enumerate(tops):
        idx = [i for i, b in enumerate(base) if b["top"] == ti]
        emit(top_id[ti], 2, None, top["name"], top["desc"], [records[i] for i in idx], wmean(idx))
    for m, md in enumerate(mids):
        idx = [i for i, b in enumerate(base) if b["mid"] == m]
        emit(
            mid_id[m],
            1,
            top_id[md["top"]],
            md["name"],
            md["description"],
            [records[i] for i in idx],
            wmean(idx),
        )
    for i, b in enumerate(base):
        emit(base_id[i], 0, mid_id[b["mid"]], b["name"], b["description"], [records[i]], base_xy[i])

    # ---------------- map points ----------------
    points = None
    if not args.no_points:
        xs, ys, cs = [], [], []
        for i in range(n_base):
            n = int(sizes[i])
            sig = base_rad[i] * 0.45
            # two overlapping lobes + a few stragglers give an irregular, UMAP-like blob
            off = rng.normal(0, sig * 0.6, 2)
            lobe = rng.random(n) < 0.6
            centre = np.where(lobe[:, None], base_xy[i] + off, base_xy[i] - off * 0.7)
            aniso = np.array([1.0, np.exp(rng.normal(0, 0.3))])
            p = centre + rng.normal(0, 1, (n, 2)) * sig * aniso
            stray = rng.random(n) < 0.04
            p[stray] += rng.normal(0, sig * 4, (stray.sum(), 2))
            xs.append(p[:, 0])
            ys.append(p[:, 1])
            cs += [base_id[i]] * n
        x, y = np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)
        perm = rng.permutation(len(x))
        points = (x[perm], y[perm], np.array(cs, dtype=object)[perm])

    # ---------------- write ----------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    con = duckdb.connect(str(args.out))
    create_schema(con)
    all_users = int(sum(r["users"] for r in records) * 0.55)
    meta = {
        "run_id": "fixture",
        "dataset": DATASET,
        "n_conversations": tot_n,
        "n_users": all_users,
        "n_clusters_base": n_base,
        "n_suppressed_clusters": 41,
        "min_users": MIN_USERS,
        "min_conversations": MIN_CONVERSATIONS,
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "date_start": "2023-04-09",
        "date_end": "2024-04-29",
        "created_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "is_fixture": "true",
    }
    con.executemany("INSERT OR REPLACE INTO run_meta VALUES (?, ?)", [(k, str(v)) for k, v in meta.items()])
    for table, rows in (("clusters", clusters), ("cluster_stats", stats), ("cluster_facets", facets)):
        insert_rows(con, table, rows)
    if points is not None:
        insert_rows(con, "map_points", list(zip(*[a.tolist() for a in points], strict=True)))
    con.close()
    print(
        f"wrote {args.out}: {len(tops)} top, {len(mids)} mid, {n_base} base clusters, "
        f"{tot_n:,} conversations, {len(facets):,} facet rows, "
        f"{0 if points is None else len(points[0]):,} map points"
    )


if __name__ == "__main__":
    main()
