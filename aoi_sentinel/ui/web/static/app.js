// Listens to /ws/inbox and renders one card per ROI event.
(() => {
  const stack = document.getElementById('card-stack');
  const tpl = document.getElementById('card-template');
  const empty = stack.querySelector('.empty');

  // htmx ws extension exposes the socket as document body's HX-WS attribute.
  // For simplicity we open our own WebSocket here and bypass the ws extension.
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/ws/inbox`);

  ws.addEventListener('message', (msg) => {
    const evt = JSON.parse(msg.data);
    if (empty && empty.parentNode === stack) empty.remove();

    const node = tpl.content.firstElementChild.cloneNode(true);
    node.dataset.board = evt.board_id;
    node.dataset.refdes = evt.ref_des;
    node.querySelector('.board').textContent = evt.board_id;
    node.querySelector('.refdes').textContent = evt.ref_des;
    node.querySelector('.vendor-call').textContent =
      `${evt.vendor}: ${evt.vendor_call}` +
      (evt.vendor_defect_type ? ` (${evt.vendor_defect_type})` : '');
    node.querySelector('.roi').src = evt.image_url;

    const form = node.querySelector('form');
    form.querySelector('[name=board_id]').value = evt.board_id;
    form.querySelector('[name=ref_des]').value = evt.ref_des;

    stack.prepend(node);
    htmx.process(node);
  });
})();
