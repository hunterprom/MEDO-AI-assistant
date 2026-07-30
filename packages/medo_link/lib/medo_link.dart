/// MEDO Link — the shared client for MEDO's companion API.
///
/// The "connect to MEDO" system, factored out of the watch app so the phone
/// (and anything else) reuses the exact same discovery + pairing + `/ask`
/// flow instead of re-implementing it.
library medo_link;

export 'src/client.dart';
export 'src/discovery.dart';
