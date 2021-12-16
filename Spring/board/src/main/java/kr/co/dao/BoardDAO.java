package kr.co.dao;

import java.util.List;

import kr.co.vo.BoardVO;
import kr.co.vo.Criteria;

public interface BoardDAO {
	// �Խñ� �ۼ�
	public void write(BoardVO boardVO) throws Exception;

	//�Խù� ��� ��ȸ
	public List<BoardVO> list(Criteria cri) throws Exception;
		
	//�Խù� �� ����
	public int listCount() throws Exception;
	
	// �Խù� ��ȸ
	public BoardVO read(int bno) throws Exception;
	
	// �Խù� ����
	public void update(BoardVO boardVO) throws Exception;
	
	// �Խù� ����
	public void delete(int bno) throws Exception;
}