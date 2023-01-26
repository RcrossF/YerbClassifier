# Dataset Acquisition
The dataset used to train the models is the entirety of Guayaki's instagram page (https://www.instagram.com/guayaki/) as of November ~30th 2022. 5922 images were scraped using [instaloader](https://instaloader.github.io/)

# Labeling
To label the dataset I used Azure's dataset labeling tool because of it's freeness and cool advertised auto-labeling feature where it generates its own models and predicts labels as you go.

That was a complete lie. While the labeling web tool works well, the ML assistance actually broke the UI and made it impossible to tag any image. This was thrown out and I went back to pure manual tagging. Overall it felt unfinished and I can't recommend anyone use it for serious tasks. For example why on earth do I have to click in the image EACH TIME to use hotkeys, and why can I only have 9 hotkeys that I can't customize whatsoever? (Class and key are chosen for you) Also for the love of god why can I not use hotkeys while zoomed in? I don't think the devs have ever actually used their website.

Anyways,

The classes are as follows:
* Cans
    * Enlightenmint
    * Revelberry
    * Bluephoria
    * Lemon
    * Other
    * Slim
        * Gold
        * Cranpom
        * Blackberry
        * Grapefruit Ginger
        * Lima Limon
        * Other
* Bottles
    * Mint
    * Raspberry
    * Original(Traditional)
    * Passion
    * Other
* No Yerb

I am missing some products (Orange, Tropical, Peach, and many limited runs to name a few) but given the small dataset and huge bias towards "No Yerb" I'll likely have to merge classes anyways.

Images were labeled with a few rules in mind:

* Any instance counts even if it's barely in frame
* Tea, stickers, etc do not count
* If I'm unsure what something is (eg. too blurry but it's definitely a bottle) I pick the base class (Bottle in this case)
* Anything that can be identified but doesn't have a class gets assigned Other (eg. a peach bottle is tagged as Bottles/Other)
* 2 different instances in the same image get tagged (eg. Mint and Lemon cans would be tagged as Cans/Enlightenmint, Cans/Lemon)

Some were pretty easy: https://i.imgur.com/MbH91SV.png
Should this count?: https://i.imgur.com/t1cE6iM.jpg (No)
This?: https://i.imgur.com/OQMGMOX.jpg (Yes, I guess...)