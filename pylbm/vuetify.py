import ipyvuetify as v
from traitlets import Unicode

class mathjax(v.Layout):

    def __init__(self, content, **kwargs):

        class myHtml(v.VuetifyTemplate):
            template = Unicode(f'''
            <template>
            <div ref="mymathjax" >
                {content}
            </div>
            </template>

            <script>
            modules.export = {{
            mounted() {{
                if (window['MathJax']) {{
                MathJax.Callback.Queue(['Typeset', MathJax.Hub, this.$refs.mymathjax])
                }}
            }}
            }}
            </script>
            ''').tag(sync=True)

        super().__init__(
            row=True,
            align_center=True,
            children=[v.Flex(xs12=True, children=[myHtml()])]
        )