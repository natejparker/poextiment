import poextiment as pxt
import plotly.express as px

def main():
    poext = pxt.Poextiment()

    poext.load_stop_words('stopwords.txt')

    poext.load_text('theraven.txt', label='The Raven')
    poext.load_text('thehauntedpalace.txt', label='The Haunted Palace')
    poext.load_text('annabellee.txt', label='Annabel Lee')
    poext.load_text('dream-land.txt', label='Dream-Land')
    poext.load_text('lenore.txt', label='Lenore')
    poext.load_text('thebells.txt', label='The Bells')
    poext.load_text('thesleeper.txt', label='The Sleeper')
    poext.load_text('ulalume.txt', label='Ulalume')
    poext.load_text('fairyland.txt', label='Fairy Land')
    poext.load_text('conquererworm.txt', label='The Conquerer Worm')

    # Various famous poems used for testing purposes
    # poext.load_text('icarryyourheartwithme.txt', label='i carry your heart... by Cummings')
    # poext.load_text('ohcaptainmycaptain.txt', label='Oh Captain!... by Whitman')
    # poext.load_text('power.txt', label='Power by Lorde')
    # poext.load_text('sonnet18.txt', label='Sonnet 18 by Shakespeare')
    # poext.load_text('stillirise.txt', label='Still I Rise by Angelou')
    # poext.load_text('theroadnottaken.txt', label='The Road Not Taken by Frost')
    # poext.load_text('thewasteland.txt', label='The Waste Land by Eliot')
    # poext.load_text('gentlegoodnight.txt', label='Do not go gentle... by Thomas')
    # poext.load_text('funeralinbrain.txt', label='I felt a Funeral... by Dickinson')

    poext.text_to_sankey(k=10)

    poext.hist_subplots()

    poext.swarm_sentiment()

    pass

main()