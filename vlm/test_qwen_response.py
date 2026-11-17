"""Response evidence is diagnostic only: malformed replies still stop immediately."""
import copy
import hashlib
import json
import unittest
from unittest.mock import Mock, patch
from vlm import qwen_review as Q
from vlm.test_qwen_review import QwenTest, response

class ResponseEvidenceTests(QwenTest):
    def choice_response(self):
        data=response()
        data['choices'][0]['message']['content']=json.dumps(dict(outcome='selected',candidate_id=7,reason='best evidence'))
        return data

    def test_distinct_validation_failures_keep_body_and_single_reservation(self):
        cases=[]
        data=self.choice_response();data['choices'][0]['message']['content']='{broken';cases.append((data,'invalid_json','candidate_choice'))
        data=self.choice_response();data['choices'][0]['finish_reason']='length';cases.append((data,'incomplete_comparison','candidate_choice'))
        data=self.choice_response();data['choices'][0]['message']['content']=json.dumps(dict(outcome='selected',candidate_id=4,reason='best'));cases.append((data,'comparison_choice','candidate_choice'))
        data=self.choice_response();data['choices'][0]['message']['content']=json.dumps(dict(outcome='selected',candidate_id=7,reason='best',extra=True));cases.append((data,'comparison_schema','candidate_choice'))
        data=self.choice_response();data['usage']['completion_tokens']=500;cases.append((data,'usage_bounds','usage'))
        data=self.choice_response();del data['usage'];cases.append((data,'invalid_response','usage'))
        for data,code,step in cases:
            with self.subTest(code=code):
                before=self.calls()
                raw=json.dumps(data)
                reply=Mock(status_code=200,json=lambda:data,text=raw)
                with patch.object(Q.requests,'post',return_value=reply) as post:
                    with self.assertRaisesRegex(RuntimeError,code):
                        self.client.compare_candidates([self.rgb]*5,'couch',[1,2,3,6,7])
                self.assertEqual(post.call_count,1)
                calls=self.calls();added=set(calls)-set(before);self.assertEqual(len(added),1)
                bill=calls[added.pop()]
                self.assertEqual(bill['response_validation_code'],code)
                self.assertEqual(bill['response_validation_step'],step)
                self.assertEqual(bill['response_body'],raw)
                self.assertEqual(bill['charged_cny'],Q.RESERVE_CNY)
                self.assertFalse(bill['retry_scheduled'])
                self.assertFalse(bill['transport']['retryable'])
                for k,v in before.items():self.assertEqual(calls[k],v)

    def test_provider_body_redacted_and_request_headers_not_persisted(self):
        data=self.choice_response();data['choices'][0]['finish_reason']='length'
        raw=json.dumps(data)+' fake-key'
        reply=Mock(status_code=200,json=lambda:data,text=raw)
        reply.headers={'Authorization':'another-sensitive-header'}
        with patch.object(Q.requests,'post',return_value=reply):
            with self.assertRaises(RuntimeError):self.client.compare_candidates([self.rgb]*2,'couch',[1,7])
        bill=list(self.calls().values())[-1]
        self.assertIn('[REDACTED]',bill['response_body'])
        self.assertNotIn('fake-key',self.ledger.read_text())
        self.assertNotIn('another-sensitive-header',self.ledger.read_text())

    def test_malformed_http_json_recorded_with_bounded_body(self):
        raw='x'*70000+'fake-key'
        reply=Mock(status_code=200,text=raw,json=Mock(side_effect=json.JSONDecodeError('bad',raw,0)))
        with patch.object(Q.requests,'post',return_value=reply):
            with self.assertRaisesRegex(RuntimeError,'invalid_json'):self.client(self.rgb,'chair')
        bill=list(self.calls().values())[-1]
        self.assertEqual(bill['response_validation_step'],'json_body')
        self.assertTrue(bill['response_body_truncated'])
        self.assertEqual(len(bill['response_body']),65536)
        self.assertEqual(bill['response_body_sha256'],hashlib.sha256(raw.replace('fake-key','[REDACTED]').encode()).hexdigest())

    def test_valid_choice_and_category_payloads_remain_unchanged(self):
        for comparison in [False,True]:
            data=self.choice_response() if comparison else response()
            with patch.object(Q.requests,'post',return_value=Mock(status_code=200,json=lambda:data,text=json.dumps(data))) as post:
                result=(self.client.compare_candidates([self.rgb]*2,'couch',[1,7]) if comparison else self.client(self.rgb,'chair'))
            self.assertEqual(result['answer'],json.loads(data['choices'][0]['message']['content']))
            self.assertNotIn('response_body',result)
            self.assertEqual(post.call_args.kwargs['json']['max_tokens'],Q.MAX_TOKENS)

if __name__=='__main__':unittest.main()
