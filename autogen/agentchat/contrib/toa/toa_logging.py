import sys

from pydantic import BaseModel


def f_text(text, indentation=0):
    # Create indentation string
    indent = "    " * indentation

    # Split the input text into words
    words = text.split()
    formatted_lines = []
    current_line = ""

    for word in words:
        # Check if adding the current word would exceed the line limit
        if len(current_line) + len(word) + 1 > 50:
            # If current_line is not empty, add it to formatted_lines
            if current_line:
                formatted_lines.append(current_line)
                current_line = ""

        # Add the word to the current line (with space if needed)
        if current_line:
            current_line += " " + word
        else:
            current_line = word

    # Add the last line if it has any content
    if current_line:
        formatted_lines.append(current_line)

    # Join lines with the specified indentation
    return ("\n" + indent).join(formatted_lines)


# This implementation of getting rid of all logs that are not my own is extremely
# messy, I should have just used a logger but I was too stubborn at the time
# I just wanted to have clean output pushed to a file with >
class CustomOutput:
    def __init__(self, original_stdout):
        # Keep a reference to the original stdout
        self.original_stdout = original_stdout

    def write(self, text):
        # Pass the output only if stdout is not redirected
        if sys.stdout == self.original_stdout:
            self.original_stdout.write(text)
            self.original_stdout.flush()

    def flush(self):
        # Required for compatibility with log and other streams
        pass

    def display(self, message):
        # Write your custom messages to the original stdout
        self.original_stdout.write(message + "\n")
        self.original_stdout.flush()


original_stdout = sys.stdout
sys.stdout = CustomOutput(original_stdout)


def log(msg, prefix="", indent_num=0):
    formatted = f_text(msg, indent_num)
    formatted = formatted.replace("\n", f"\n{str(prefix)}")
    sys.stdout.display(str(prefix) + (indent_num * "    ") + formatted)


def log_sentence_result(sentence_result, prefix, indent):
    (grade, grade_justification, fact, fact_memory, sentence_agent, paragraph_name) = sentence_result

    log("Fact:", prefix, indent)
    log(f_text(fact, indent + 2), prefix, indent + 1)

    log("Grade:", prefix, indent)
    log(str(grade), prefix, indent + 1)

    log("Grade Justification:", prefix, indent)
    log(f_text(grade_justification, indent + 2), prefix, indent + 1)

    log("Source Paragraph Name:", prefix, indent)
    log(paragraph_name, prefix, indent + 1)

    log("Source Sentence:", prefix, indent)
    log(f_text(sentence_agent.sentence, indent + 2), prefix, indent + 1)


def log_sub_questions(sub_questions):
    log("Sub Questions:", indent_num=2)
    for i, sub_question in enumerate(sub_questions):
        log(f"{i}:", i, 1)
        log(f"{sub_question}", i, 2)


class Level(BaseModel):
    level_num: int
    question: str
    answer: str
    confidence: int
    confidence_justification: str
    answer_justification: str
    sentence_results: list

    def log_level(self, title="Level", log_sentence_stuff=True):
        p = self.level_num + 1
        index = 1
        log(f"{title} {self.level_num}", p)

        log("Question:", p, index)
        log(f_text(self.question, index + 1), p, index + 1)

        log("Answer:", p, index)
        log(f_text(self.answer, index + 1), p, index + 1)

        log("Confidence:", p, index)
        log(str(self.confidence), p, index + 1)

        log("Confidence Justification:", p, index)
        log(f_text(self.confidence_justification, index + 1), p, index + 1)

        log("Answer Justification:", p, index)
        log(f_text(self.answer_justification, index + 1), p, index + 1)

        if log_sentence_stuff:
            for i in range(len(self.sentence_results)):
                log_sentence_result(self.sentence_results[i], self.level_num, index + 2)


def log_beam_answer(beam_answer, title="Level "):
    (
        answer,
        answer_justification,
        confidence,
        confidence_justification,
        sub_question,
        all_sentence_results,
        beam_memories,
        level_num,
    ) = beam_answer
    level = Level(
        level_num=level_num,
        question=sub_question,
        answer=answer,
        confidence=confidence,
        confidence_justification=confidence_justification,
        answer_justification=answer_justification,
        sentence_results=all_sentence_results,
    )
    level.log_level(log_sentence_stuff=False, title=title)


def log_beam_answers(beam_answers):
    log("#### Beam Results ####")
    for beam_answer in beam_answers:
        log_beam_answer(beam_answer, "Beam")
    log("## End Beam Results ##")


def log_levels(levels):
    for level in levels:
        level.log_level(log_sentence_stuff=False)
