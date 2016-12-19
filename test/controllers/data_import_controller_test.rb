require 'test_helper'

class DataImportControllerTest < ActionController::TestCase
  setup do
    session[:logged_in] = true
  end

  test "get index" do
    get :index
    assert_response :success
  end

  test "send file" do
    # TODO: cant find tempfile in test
    post :create, params: { exchange_rate_file: { tempfile: File.open(File.join(Rails.root, 'test','fixtures', 'files', 'exchange_rates', 'valid_exchange_rates.csv')) } }
    assert_redirected_to data_import_path
  end
end