from src.lib import utils


def main():
    # ATTENTION: if you want to use LLM,
    # be sure you have applied the key of baidu api and correct utils.py

    # Get ST code from LLM API
    # llm = utils.ErnieAPI()
    # llm.ask('Generate an ST code for a PLC controlled motor')

    # Build the AST and python code of ST code
    interpreter = utils.Interpreter('output/st-code.ST')
    interpreter.interpret()

    # Unroll the function in python code
    unroller = utils.Unroller('output/python-code.py')
    unroller.unroll()

    # Build XMV model
    model_builder = utils.PyNuXMV()
    model_builder.build()


if __name__ == '__main__':
    main()
