# %%
import re
from flashtext import KeywordProcessor


# %%


def remove_words_flashtext(text: str, words_to_remove: list[str]) -> str:
    keyword_processor = KeywordProcessor()
    for word in words_to_remove:
        keyword_processor.add_keyword(word, "__")
    replaced_with_underscores = keyword_processor.replace_keywords(text)
    return (
        replaced_with_underscores.replace("__, ", "")
        .replace("__,", "")
        .replace(",__", "")
        .replace(" __", "")
        .replace("__ ", "")
        .replace("__", "")
    )


yt_text = "[Music]Um, hi I'm okay and I like um uh this stuff: Cheese, hm,eggs,um. in this video I'm going to show you how to use our air probe features of our AP firmware uh we used to have the air probes. and now we can just tell an any AP to become an air probe whenever you tell an AP to become an air probe it will no longer act as an access point so all you have to do is pull up your AP click edit scroll down enable air monitor mode then click save save once you do that um I I it'll change to air monitor mode right there and so you'll see you're in air monitor mode once again now that we've done this any Wi-Fi access point or Wi-Fi way on this R will not operate this guy is only designed to be a a prob so then I can hit settings remote web admin log into the device and you'll see there's no AP tab here anymore so I can go to the system tab go to Wi-Fi air monitoring performance test and you can create monitors for different SSID so here you can program your SSID settings and then you can put your test schedule want it to be every hour every each day every Monday Tuesday every day so you can say every day at 8:00 a.m. and then run a speed test and then hit save for the purpose of this video I I went ahead and turned my upstairs uh AP into an air Monitor and then I added it to to run scans basically every 15 minutes and then it's Shing us speed test on my 5 GHz Channel and then it experiments the 2.4 at the 55 minute marker of every hour so once I've done that I hit save and apply changes and come back to my air probe and then I can go to reports and I've got air Monitor and performance test so if I pull up performance test I can pull up these reports that I've run see 12:45 1 115 and 130 so I can and and so that's running on my keer West 5 gz and then here's my ker West 2.4 GHz test so I can pull up my 5 GHz test click on one of these and I can see all the details about that test what I find really neat too though is if you go to reports and then air monitor I can you can select a Time click on either 2.4 or 5 GHz and see what's happening at that time so I can you know scroll it down and change it you can see report time uh at 10:04 p.m. I can see my my network utilization I can see my total unique nodes you see so you you get all these datas um RSSI the channels what's happening on your AP distribution and so you can pull up information about how this network is operating it basically gives you remote Spectrum analyzer like right there you can see all your channels what's happening what's around you see how we can see everything and then you can see they're overlapping on the channels in the 2.4 gahz range and then I can go to my 5 gahz range and see my channel over overhead right there so you can see those you can see it's like some sort of HP scanner and Camera um and then here you can see AP station station probe and unknown and you can see what what frequ what channel they're all operating on and so kind of gives you some some visibility there um and then once again you can see network utilization which right now is little just wanted to give you some visibility into this and how to turn it on"
DEFAULT_DISFLUENCIES = {
    "um",
    "uh",
    "uhm",
    "ok",
    "okay",
    "o.k.",
    "hmm",
    "hm",
    "well",
    "so",
    "I'm",
    "cheese",
}


yt_text = yt_text.replace("[Music]", "")
yt_text = yt_text.replace("\n", " ")


yt_result = remove_words_flashtext(yt_text, DEFAULT_DISFLUENCIES)
print(yt_result)
