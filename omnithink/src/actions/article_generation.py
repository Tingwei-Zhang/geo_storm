import concurrent.futures
import copy
import logging
import random
from concurrent.futures import as_completed
from typing import List, Union
import random
import dspy
import sys
from src.utils.ArticleTextProcessing import ArticleTextProcessing



# This code is originally sourced from Repository STORM
# URL: [https://github.com/stanford-oval/storm]
class ArticleGenerationModule():
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage, 
    """

    def __init__(self,
                 retriever,
                 article_gen_lm=Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 retrieve_top_k: int = 10,
                 max_thread_num: int = 10,
                 agent_name: str = 'WriteSection' ,
                 ):
        super().__init__()
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.max_thread_num = max_thread_num
        self.retriever = retriever
        self.section_gen = ConvToSection(engine=self.article_gen_lm, class_name=agent_name)

    def generate_section(self, topic, section_name, mindmap, section_query, section_outline, language_style):
        collected_info = mindmap.retrieve_information(queries=section_query,
                                                      search_top_k=self.retrieve_top_k)
        output = self.section_gen(
            topic=topic,
            outline=section_outline,
            section=section_name,
            collected_info=collected_info,
            language_style=language_style,
        )

        return {"section_name": section_name, "section_content": output.section, "collected_info": collected_info}

    def generate_article(self,
                         topic: str,
                         mindmap,
                         article_with_outline,
                         language_style=None,
                         ):
        """
        Generate article for the topic based on the information table and article outline.
        """
        mindmap.prepare_table_for_retrieval()
        language_style = "{} {}\n".format(language_style.get('style', ''),
                                          language_style.get('language_type', '')) if language_style else str()

        sections_to_write = article_with_outline.get_first_level_section_names()
        section_output_dict_collection = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
            future_to_sec_title = {}
            for section_title in sections_to_write:
                section_query = article_with_outline.get_outline_as_list(
                    root_section_name=section_title, add_hashtags=False
                )
                queries_with_hashtags = article_with_outline.get_outline_as_list(
                    root_section_name=section_title, add_hashtags=True
                )
                section_outline = "\n".join(queries_with_hashtags)

                future_to_sec_title[
                    executor.submit(self.generate_section,
                                    topic, section_title, mindmap, section_query, section_outline, language_style)
                ] = section_title

            for future in concurrent.futures.as_completed(future_to_sec_title):
                section_output_dict_collection.append(future.result())

        article = copy.deepcopy(article_with_outline)
        for section_output_dict in section_output_dict_collection:
            article.update_section(parent_section_name=topic,
                                   current_section_content=section_output_dict["section_content"],
                                   current_section_info_list=section_output_dict["collected_info"],
                                   )

        article.post_processing()

        return article


class ConvToSection(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""

    def __init__(self, class_name, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        # self.write_section = dspy.Predict(WriteSection)
        current_module = globals()
        if class_name in current_module:
            cls = current_module.get(class_name)
            self.write_section = dspy.Predict(cls)
        else:
            raise ValueError(f"Class '{class_name}' not found!")
        self.engine = engine

    def forward(self, topic: str, outline: str, section: str, collected_info: List, language_style: str):
        all_info = ''
        for idx, info in enumerate(collected_info):
            all_info += f'[{idx + 1}]\n' + '\n'.join(info['snippets'])
            all_info += '\n\n'

            all_info = ArticleTextProcessing.limit_word_count_preserve_newline(all_info, 1500)

            with dspy.settings.context(lm=self.engine):
                section = ArticleTextProcessing.clean_up_section(
                    self.write_section(topic=topic, info=info, section=section, language_style=language_style).output)

        section = section.replace('\[', '[').replace('\]', ']')
        return dspy.Prediction(section=section)


class WriteSection(dspy.Signature):
    """Write a Wikipedia section based on the collected information.

    Here is the format of your writing:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
        3. The language style should resemble that of Wikipedia: concise yet informative, formal yet accessible.
    """
    info = dspy.InputField(prefix="The Collected information:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    section = dspy.InputField(prefix="The section you need to write: ", format=str)
    language_style = dspy.InputField(prefix='the language style you needs to imitate: ', format=str)
    output = dspy.OutputField(
        prefix="Write the section with proper inline citations (Start your writing with # section title. Don't include the page title or try to write other sections):\n",
        format=str)


class WriteSectionAgentEnglish(dspy.Signature):
    """Generate an English Wikipedia section with formal yet accessible tone, adhering to standard Wikipedia guidelines.

    Writing specifications:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3].").
        3. Maintain formal yet accessible English language style
        4. Follow standard English Wikipedia formatting and style guidelines
        5. Ensure clarity and readability for international English readers
    """
    info = dspy.InputField(prefix="Collected source materials:\n", format=str)
    topic = dspy.InputField(prefix="Article topic: ", format=str)
    section = dspy.InputField(prefix="Target section to write: ", format=str)
    language_style = dspy.InputField(prefix='Target writing style (formal English): ', format=str)
    output = dspy.OutputField(
        prefix="Generate the English section with proper inline citations (start with # section title, exclude page header):\n",
        format=str)


class WriteSectionAgentChinese(dspy.Signature):
    """Generate a Chinese Wikipedia section adhering to standard formatting and style guidelines.

    Writing specifications:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3].").
        3. Maintain formal yet accessible Chinese language style
        4. Follow standard Chinese Wikipedia formatting and style guidelines
        5. Use Simplified Chinese characters and proper punctuation
    """
    info = dspy.InputField(prefix="Collected source materials:\n", format=str)
    topic = dspy.InputField(prefix="Article topic: ", format=str)
    section = dspy.InputField(prefix="Target section to write: ", format=str)
    language_style = dspy.InputField(prefix='Target writing style (formal Chinese): ', format=str)
    output = dspy.OutputField(
        prefix="Generate the Chinese section with proper inline citations (start with # section title, exclude page header):\n",
        format=str)


class WriteSectionAgentFormalChinese(dspy.Signature):
    """Generate a formal Chinese Wikipedia section based on the collected information.

    Writing specifications:
    1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
    2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3].").
    3. Maintain formal and professional Chinese language style
    4. Follow Wikipedia's neutral tone and encyclopedic writing standards
    5. Use standard Simplified Chinese characters and proper punctuation
    """
    info = dspy.InputField(prefix="Collected source materials:\n", format=str)
    topic = dspy.InputField(prefix="Article topic: ", format=str)
    section = dspy.InputField(prefix="Target section to write: ", format=str)
    language_style = dspy.InputField(prefix='Target writing style (formal Chinese): ', format=str)
    output = dspy.OutputField(
        prefix="Generate the formal Chinese section with proper inline citations (start with # section title, exclude table of contents):\n",
        format=str)


class WriteSectionAgentEnthusiasticChinese(dspy.Signature):
    """Generate an engaging Chinese Wikipedia section with enthusiastic tone while maintaining factual accuracy.

    Writing specifications:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3].").
        3. Employ lively yet professional Chinese language style
        4. Maintain Wikipedia's neutral point of view while using engaging expressions
        5. Use appropriate rhetorical devices to enhance readability
    """
    info = dspy.InputField(prefix="Collected source materials:\n", format=str)
    topic = dspy.InputField(prefix="Article topic: ", format=str)
    section = dspy.InputField(prefix="Target section to write: ", format=str)
    language_style = dspy.InputField(prefix='Target writing style (enthusiastic Chinese): ', format=str)
    output = dspy.OutputField(
        prefix="Generate the enthusiastic Chinese section with proper inline citations (start with # section title, maintain engaging tone):\n",
        format=str)


class WriteSectionAgentEnthusiasticEnglish(dspy.Signature):
    """Generate an engaging English Wikipedia section with enthusiastic tone while maintaining factual accuracy.

    Writing specifications:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3].").
        3. Employ lively yet professional English language style
        4. Maintain Wikipedia's neutral point of view while using engaging expressions
        5. Use appropriate rhetorical devices to enhance readability
        6. Follow standard English Wikipedia formatting guidelines
    """
    info = dspy.InputField(prefix="Collected source materials:\n", format=str)
    topic = dspy.InputField(prefix="Article topic: ", format=str)
    section = dspy.InputField(prefix="Target section to write: ", format=str)
    language_style = dspy.InputField(prefix='Target writing style (enthusiastic English): ', format=str)
    output = dspy.OutputField(
        prefix="Generate the enthusiastic English section with proper inline citations (start with # section title, maintain engaging tone):\n",
        format=str)


class WriteSectionAgentFormalEnglish(dspy.Signature):
    """Generate a formal English Wikipedia section adhering to standard formatting and style guidelines.

    Writing specifications:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3].").
        3. Maintain formal and professional English language style
        4. Follow standard English Wikipedia formatting and style guidelines
        5. Ensure clarity and readability for international English readers
    """
    info = dspy.InputField(prefix="Collected source materials:\n", format=str)
    topic = dspy.InputField(prefix="Article topic: ", format=str)
    section = dspy.InputField(prefix="Target section to write: ", format=str)
    language_style = dspy.InputField(prefix='Target writing style (formal English): ', format=str)
    output = dspy.OutputField(
        prefix="Generate the formal English section with proper inline citations (start with # section title, exclude page header):\n",
        format=str)