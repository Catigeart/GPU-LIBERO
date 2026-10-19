"""Retained BoxBox source correction, also checked against the CUDA body.

This helper does not regenerate or replace production CUDA sources.
"""


def correct_boxbox_edge_contacts(body):
    """Keep lateral feature gaps out of signed penetration depths.

    The no-intersection fallback clamps corners to the other face. A clamped
    point outside that face has a lateral gap, even when it is below the
    reference plane. Its Euclidean distance is positive and must fit margin.
    Keep the projection determinant intact across all four corner queries.
    """
    replacements = [
        ('      if (nl == 0) {',
         '      const int outside = u < 0 || u > 1 || v < 0 || v > 1;\n'
         '      if (nl == 0) {'),
        ('      c1 = mju_dot3(tmp2, tmp2);\n'
         '      if (tmp1[2] > 0)\n'
         '        if (c1 > margin2)',
         '      // c1 remains the projection determinant for the next corner.\n'
         '      c2 = mju_dot3(tmp2, tmp2);\n'
         '      if (outside || tmp1[2] > 0)\n'
         '        if (c2 > margin2)'),
        ('      depth[n] = sqrt(c1) * (tmp1[2] < 0 ? -1 : 1);',
         '      depth[n] = sqrt(c2) * (!outside && tmp1[2] < 0 ? -1 : 1);'),
        ('      x = ppts2[i][0];\n      y = ppts2[i][1];',
         '      x = ppts2[i][0];\n      y = ppts2[i][1];\n'
         '      const int outside = x < -lx || x > lx || y < -ly || y > ly;'),
        ('      if (pu[i][2] > 0)\n        if (c1 > margin2)',
         '      if (outside || pu[i][2] > 0)\n        if (c1 > margin2)'),
        ('      depth[n] = sqrt(c1) * (pu[i][2] < 0 ? -1 : 1);',
         '      depth[n] = sqrt(c1) * (!outside && pu[i][2] < 0 ? -1 : 1);'),
    ]
    # Restrict the first replacement to the corner-projection loop; a second
    # no-intersection branch below has a different projected-footprint test.
    head, tail = body.split('edgeedge:', 1)
    for old, new in replacements:
        if old not in tail:
            raise ValueError('BoxBox edge-contact source drift: ' + old)
        tail = tail.replace(old, new, 1)
    return head + 'edgeedge:' + tail
