<Document_1>

## Post

### Title: My connection IS SLOW

### Content:

I connected a 50mps fibre optic internet connection to port 1 and a 10mbps phone internet service to port two.
Then cabled from Lan 1 to a wi fi router and I connect my devices via wi fi to that router.
However this is slower and more unstable than if I just connect my 50mps straight to my wifi router, overriding the Peplink.
I can see this when I test on Speedtest.net.

## Comments:

<comment>
Speedtest.net does not give an accurate reading when using multiple WANs because it doesn't take the Pepwave's load balancing into account. You must configure outbound policy rules in order to implement your scenario. Set your outbound policy to use the low-latency option and it will use the fastest connection.
  <reply>
  Hi Ron!
Does this mean that if I am using Strong DNS, this might be slowing it down?
    <reply>
    No, you just need to configure outbound policy rules.
    </reply>
  </reply>
</comment>

</Document_1>

<Document_2>

hi this is Dan and in this video I want to explain how to configure the outbound policy rules there are several options available and they all have different use cases when in doubt the best option is the power-fusion option that will use the fastest connection based on the FQDN protocol it is the best option to use for most simple use cases or if you are unsure I would recommend taking a look at the manual to learn about the other options available to you

</Document_2>

<Document_3>

You know about classes, but you may be thinking which traffic goes into each class? And how many classes should I configure? Well, let me help you simplify this You don't need to use every combination of class and drop probability. There is a real time class for voice and interactive video. This is a high priority class next, a class for critical data. This is for your business applications, databases, website, traffic. We could split this into two classes. If you do end up splitting your high priority class into two classes, make sure you set the outbound policy to use low-latency otherwise your equipment will not be able to handle the load and you may encounter a "unable to handle load" error. I would recommend starting with the default settings and then adding classes as needed. In all classes are not as complex as they may seem.

</Document_3>