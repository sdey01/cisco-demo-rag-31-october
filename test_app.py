import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.documents import Document

from app import answer_question, render_history


class AnswerQuestionTests(unittest.TestCase):
    def test_builds_grounded_prompt_and_limits_results_to_five(self):
        documents = [
            Document(page_content=f"Guide passage {index}", metadata={"page": index})
            for index in range(6)
        ]
        retriever = Mock()
        retriever.invoke.return_value = documents
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content="Grounded answer.")

        answer, retrieved = answer_question("  What does the guide say?  ", retriever, llm)

        self.assertEqual(answer, "Grounded answer.")
        self.assertEqual(len(retrieved), 5)
        retriever.invoke.assert_called_once_with("What does the guide say?")
        prompt = llm.invoke.call_args.args[0]
        self.assertIn("[Page 1]\nGuide passage 0", prompt)
        self.assertIn("What does the guide say?", prompt)
        self.assertNotIn("Guide passage 5", prompt)

    def test_no_retrieved_documents_returns_transparent_answer(self):
        retriever = Mock()
        retriever.invoke.return_value = []
        llm = Mock()

        answer, retrieved = answer_question("What does the guide say?", retriever, llm)

        self.assertIn("does not provide sufficient information", answer)
        self.assertEqual(retrieved, [])
        llm.invoke.assert_not_called()

    def test_empty_question_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Enter a question"):
            answer_question("  ", Mock(), Mock())


class HistoryTests(unittest.TestCase):
    def test_empty_history_is_empty(self):
        self.assertEqual(render_history([]), [])

    def test_history_renders_recent_question_answer_pairs(self):
        entries = [
            {"question": f"Question {index}", "answer": f"Answer {index}"}
            for index in range(12)
        ]
        rendered = render_history(entries)

        self.assertEqual(len(rendered), 20)
        self.assertEqual(rendered[0], {"role": "user", "content": "Question 2"})
        self.assertEqual(rendered[-1], {"role": "assistant", "content": "Answer 11"})


if __name__ == "__main__":
    unittest.main()
