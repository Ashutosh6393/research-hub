import sys

from pipeline.pipeline import run_research_pipeline

sys.stdout.reconfigure(errors="replace")


def main():
    print("input the topic of the research: ")
    topic = input()
    run_research_pipeline(topic)

if __name__ == "__main__":
    main()
