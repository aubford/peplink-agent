## Document_1

The Power-Fusion outbound policy rule uses the fastest connection based on the Fully Qualified Domain Name (FQDN) protocol and is recommended for most simple use cases. When configuring outbound policy rules on Pepwave devices, multiple options are available, each suited for different scenarios, and consulting the manual is advised to understand all available configurations. Power-Fusion was added in version 6.7. If a connection is slower than expected, this is the first thing you should check.

## Document_2

To troubleshoot a slow connection when using multiple physical connections and Strong DNS, try configuring outbound policy rules to use the "low-latency" option. This ensures that the Pepwave device routes traffic through the fastest available WAN connection. The use of services like Strong DNS does not inherently cause slowdowns, so try changing outbound policy rules first. Note that Speedtest.net is not a reliable tool for assessing performance when using multiple WAN connections on a Pepwave device because it does not account for the effects of load balancing.

## Document_3

Traffic should be assigned to classes based on application priority, with a designated real-time class for voice and interactive video as the highest priority. Critical business applications such as databases and websites should be placed in a separate high-priority traffic class, which can optionally be split into two distinct classes. When splitting your connection into more than one class, the outbound policy must be configured to use the low-latency setting to prevent "unable to handle load" errors which would result in a sluggish connection. It is advisable to begin with default class configurations and only introduce additional classes as operational requirements evolve.