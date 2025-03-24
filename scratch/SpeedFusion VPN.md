SpeedFusion VPN

To configure SpeedFusion VPN, navigate to Advanced > SpeedFusion VPN . The local LAN subnet and subnets behind the LAN (defined under Static Route on the LAN settings page) will be advertised to the VPN. All VPN members (branch offices and headquarters) will be able to route to local subnets. Note that all LAN subnets and the subnets behind them must be unique. Otherwise, VPN members will not be able to access each other. All data can be routed over the VPN using the 256-bit AES encryption standard. To configure, navigate to Advanced > SpeedFusion VPN and click the New Profile button to create a new VPN profile (you may have to first save the displayed default profile in order to access the New Profile button). Each profile specifies the settings for making VPN connection with one remote Pepwave or Peplink device. Note that available settings vary by model. A list of defined SpeedFusion connection profiles and a Link Failure Detection Time option will be shown. Click the New Profile button to create a new VPN connection profile for making a VPN connection to a remote Peplink MAX via the available WAN connections. Each profile is for making a VPN connection with one remote Peplink MAX. <table><tr><td>SpeedFusion VPN Profile</td></tr> <tr><td>Name</td> <td>This field is for specifying a name to represent this profile. The
                      name can be any combination of alphanumeric characters (0-9, A-Z,
                      a-z), underscores (_), dashes (-), and/or non-leading/trailing
                      spaces ( ). Click the icon next to the SpeedFusion VPN Profile title bar to use the IP ToS field of your data packet on
                        SpeedFusion VPN WAN traffic.</td></tr> <tr><td>Enable</td> <td>When this box is checked, this VPN connection profile will be
                      enabled. Otherwise, it will be disabled.</td></tr> <tr><td>Encryption</td> <td>By default, VPN traffic is encrypted with 256-bit AES . If Off is selected on both sides of a VPN connection, no encryption will be
                      applied.</td></tr> <tr><td>Authentication</td> <td>Select from By Remote ID Only , Preshared Key , or X.509 to specify the method the Peplink MAX will use to authenticate
                      peers. When selecting By Remote ID Only , be sure to enter a unique peer ID number in the Remote ID field.</td></tr> <tr><td>Remote ID / Pre-shared Key</td> <td>This optional field becomes available when Remote ID / Pre-shared Key is selected as the Peplink MAX’s VPN Authentication method, as explained above. Pre-shared Key defines the pre-shared key used for this particular VPN connection.
                      The VPN connection’s session key will be further protected by the
                      pre-shared key. The connection will be up only if the pre-shared
                      keys on each side match. When the peer is running firmware 5.0+,
                      this setting will be ignored. Enter Remote IDs either by typing out each Remote ID and
                        Pre-shared Key, or by pasting a CSV. If you wish to paste a CSV,
                        click the icon next to the “Remote ID / Preshared Key” setting.</td></tr> <tr><td>Remote ID/Remote Certificate</td> <td>These optional fields become available when X.509 is selected as the Peplink MAX’s VPN authentication method, as
                      explained above. To authenticate VPN connections using X.509
                      certificates, copy and paste certificate details into these fields.
                      To get more information on a listed X.509 certificate, click the Show Details link below the field.</td></tr> <tr><td>Allow Shared Remote ID</td> <td>When this option is enabled, the router will allow multiple peers
                      to run using the same remote ID.</td></tr> <tr><td>NAT Mode</td> <td>Check this box to allow the local DHCP server to assign an IP
                      address to the remote peer. When NAT Mode is enabled, all remote traffic over the VPN will be tagged with the
                      assigned IP address using network address translation.</td></tr> <tr><td>Remote IP Address / Host Names (Optional)</td> <td>If NAT Mode is not enabled, you can enter a remote peer’s WAN IP address or
                      hostname(s) here. If the remote uses more than one address, enter
                      only one of them here. Multiple hostnames are allowed and can be
                      separated by a space character or carriage return. Dynamic-DNS host
                      names are also accepted. This field is optional. With this field filled, the Peplink MAX
                        will initiate connection to each of the remote IP addresses until
                        it succeeds in making a connection. If the field is empty, the
                        Peplink MAX will wait for connection from the remote peer.
                        Therefore, at least one of the two VPN peers must specify this
                        value. Otherwise, VPN connections cannot be established. Click the icon to customize the handshake port of the remote
                        Host  (TCP)</td></tr> <tr><td>Cost</td> <td>Define path cost for this profile. OSPF will determine the best route through the network using the
                        assigned cost. Default: 10</td></tr> <tr><td>Data Port</td> <td>This field is used to specify a UDP port number for transporting
                      outgoing VPN data. If Default is selected, UDP port 4500 will be used. Port 32015 will be used if
                      the remote unit uses Firmware prior to version 5.4 or if port 4500
                      is unavailable. If Custom is selected, enter an outgoing port number from 1 to 65535. Click the icon to configure data stream using TCP protocol
                        [EXPERIMENTAL]. In the case TCP protocol is used, the exposed TCP
                        session option can be authorised to work with TCP accelerated WAN
                        link.</td></tr> <tr><td>Bandwidth Limit</td> <td>Define maximum download and upload speed to each individual peer.
                      This functionality requires the peer to use PepVPN version 4.0.0 or
                      above.</td></tr> <tr><td>TCP Ramp Up</td> <td>For every new TCP connection, Normal WAN Smoothing will be applied
                      for a short period of time to prevent packet loss during TCP Slow
                      Start, which in some conditions will ramp up TCP throughput
                      faster.</td></tr> <tr><td>WAN Smoothing</td> <td>While using PepVPN, utilize multiple WAN links to reduce the impact
                      of packet loss and get the lowest possible latency at the expense of
                      extra bandwidth consumption. This is suitable for streaming
                      applications where the average bitrate requirement is much lower
                      than the WAN’s available bandwidth. Off – Disable WAN Smoothing. Normal – The total bandwidth consumption will be at most 2x of
                        the original data traffic. Medium – The total bandwidth consumption will be at most 3x of
                        the original data traffic. High – The total bandwidth consumption depends on the number of
                        connected active tunnels.</td></tr> <tr><td>Forward Error Correction</td> <td>Forward Error Correction (FEC) can help to recover packet loss by
                      using extra bandwidth to send redundant data packets. Higher FEC
                      level will recover packets on a higher loss rate link. For more information on FEC and Adaptive FEC, refer to this KB article . Require peer using PepVPN version 8.0.0 and above.</td></tr> <tr><td>Receive Buffer</td> <td>Receive Buffer can help to reduce out-of-order packets and jitter,
                      but will introduce extra latency to the tunnel. Default is 0 ms,
                      which disables the buffer, and maximum buffer size is 2000 ms.</td></tr> <tr><td>Packet Fragmentation</td> <td>If the packet size is larger than the tunnel’s MTU, it will be
                      fragmented inside the tunnel in order to pass through. Select Always to fragment any packets that are too large to send,
                        or Use DF Flag to only fragment packets with Don’t Fragment bit
                        cleared. This can be useful if your application does Path MTU
                        Discovery, usually sending large packets with DF bit set, if
                        allowing them to go through by fragmentation, the MTU will not be
                        detected correctly.</td></tr> <tr><td>Use IP ToS ^</td> <td>If Use IP ToS is enabled, the ToS value of the data packets will be
                      copied to the PepVPN header during encapsulation.</td></tr> <tr><td>Latency Difference Cutoff ^</td> <td>Traffic will be stopped for links that exceed the specified
                      millisecond value with respect to the lowest latency link. (e.g.
                      Lowest latency is 100ms, a value of 500ms means links with latency
                      600ms or more will not be used)</td></tr></table> ^ Advanced feature, please click the button on the top right-hand corner to activate. To enable Layer 2 Bridging between SpeedFusion VPN profiles, navigate to Network > LAN > Basic Settings > *LAN Profile Name* and refer to instructions in section 8.1 <table><tr><td>Traffic Distribution</td></tr> <tr><td>Policy</td> <td>This option allows you to select the desired out-bound traffic
                      distribution policy: <li>Bonding – Aggregate multiple WAN-to-WAN links into a single
                          higher throughput tunnel.</li> <li>Dynamic Weighted Bonding – Aggregates WAN-to-WAN links with
                          similar latencies.</li> By default, Bonding is selected as a traffic distribution
                        policy.</td></tr> <tr><td>Congestion Latency Level</td> <td>For most WANs, especially on cellular networks, the latency will
                      increase when the link becomes more congested. Setting the Congestion Latency Level to Low will treat the link as congested more aggressively. Setting it to High will allow the latency to increase more before treating it as
                        congested.</td></tr> <tr><td>Ignore Packet Loss Event</td> <td>By default, when there is packet loss, it is considered as a
                      congestion event. If this is not the case, select this option to
                      ignore the packet loss event.</td></tr> <tr><td>Disable Bufferbloat Handling</td> <td>Bufferbloat is a phenomenon on the WAN side when it is congested.
                      The latency can become very high due to buffering on the uplink. By
                      default, the Dynamic Weighted Bonding policy will try its best to
                      mitigate bufferbloat by reducing TCP throughput when the WAN is
                      congested. However, as a side effect, the tunnel might not achieve
                      maximum bandwidth. Selecting this option will disable the bufferbloat handling mentioned above.</td></tr> <tr><td>Disable TCP ACK Optimization</td> <td>By default, TCP ACK will be forwarded to remote peers as fast as
                      possible. This will consume more bandwidth, but may help to improve
                      TCP performance as well. Selecting this option will disable the TCP ACK optimization mentioned above.</td></tr> <tr><td>Packet Jitter Buffer</td> <td>The default jitter buffer is 150ms, and can be modified from 0ms to
                      500ms. The jitter buffer may increase the tunnel latency. If you
                      want to keep the latency as low as possible, you can set it to 0ms
                      to disable the buffer. Note : If the Receive Buffer is set, the Packet Jitter Buffer will be
                        automatically disabled.</td></tr></table> <table><tr><td>WAN Connection Priority</td></tr> <tr><td>WAN Connection Priority</td> <td>If your device supports it, you can specify the priority of WAN
                      connections to be used for making VPN connections. WAN connections
                      set to OFF will never be used. Only available WAN connections with the highest
                      priority will be used. To enable asymmetric connections, connection mapping to remote
                        WANs, cut-off latency, and packet loss suspension time, click the button.</td></tr></table> <table><tr><td>Send All Traffic To</td></tr> <tr><td>This feature allows you to redirect all traffic to a specified
                      SpeedFusion VPN connection. Click the button to select your connection and the following menu will
                      appear: You could also specify a DNS server to resolve incoming DNS
                        requests. Click the checkbox next to Backup Site to designate a backup SpeedFusion profile that will take over,
                        should the main SpeedFusion VPN connection fail.</td></tr></table> <table><tr><td>Outbound Policy/SpeedFusion VPN Outbound Custom Rules</td></tr> <tr><td>Some models allow you to set outbound policy and custom outbound
                      rules from Advanced>SpeedFusion VPN . See Section 14 for more information on outbound policy settings.</td></tr></table> <table><tr><td>SpeedFusion VPN Local ID</td></tr> <tr><td>The local ID is a text string to identify this local unit when
                      establishing a VPN connection. When creating a profile on a remote
                      unit, this local ID must be entered in the remote unit’s Remote ID field. Click the icon to edit Local ID .</td></tr></table> <table><tr><td>Link Failure Detection Settings</td></tr> <tr><td>Cold Failover Mode</td> <td>Enabling Cold Failover Mode turns off link failure detection for
                      standby WAN-to-WAN links, helping to lower bandwidth
                      consumption.</td></tr> <tr><td>Link Failure Detection Time</td> <td>The bonded VPN can detect routing failures on the path between two
                      sites over each WAN connection. Failed WAN connections will not be
                      used to route VPN traffic. Health check packets are sent to the
                      remote unit to detect any failure. The more frequently checks are
                      sent, the shorter the detection time, although more bandwidth will
                      be consumed. When Recommended (default) is selected, a health check packet is sent every five
                        seconds, and the expected detection time is 15 seconds. When Fast is selected, a health check packet is sent every three seconds,
                        and the expected detection time is six seconds. When Faster is selected, a health check packet is sent every second, and the
                        expected detection time is two seconds. When Extreme is selected, a health check packet is sent every 0.1 second, and
                        the expected detection time is less than one second.</td></tr></table> ^ Advanced feature, please click the button on the top right-hand corner to activate. <table><tr><td>Important Note</td></tr> <tr><td>Peplink proprietary SpeedFusion TM uses TCP port 32015 and UDP port 4500 for establishing VPN
                      connections. If you have a firewall in front of your Pepwave
                      devices, you will need to add firewall rules for these ports and
                      protocols to allow inbound and outbound traffic to pass through the
                      firewall.</td></tr></table> <table><tr><td>Tip</td></tr> <tr><td>Want to know more about VPN sub-second session failover? Visit our YouTube Channel for a video tutorial! http://youtu.be/TLQgdpPSY88</td></tr></table>